import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

class YoloAugmentor:
    """
    Clase para aplicar Data Augmentation a imágenes y etiquetas en formato YOLO.
    """
    def __init__(self, original_img_dir, original_label_dir, augmented_img_dir, augmented_label_dir):
        # Rutas del proyecto
        self.original_img_dir = original_img_dir
        self.original_label_dir = original_label_dir
        self.augmented_img_dir = augmented_img_dir
        self.augmented_label_dir = augmented_label_dir

        # Obtener la lista de archivos de imagen (solo JPG)
        self.image_files = sorted(glob(os.path.join(original_img_dir, '*.jpg')))
        
        # Inicializar el pipeline de transformaciones de Albumentations
        self.transform = self._create_augmentation_pipeline()

        print(f"Data Augmentation iniciado. Encontradas {len(self.image_files)} imágenes.")
        
    def _create_augmentation_pipeline(self):
        """
        Define el pipeline de transformaciones de Albumentations.
        Implementa las transformaciones solicitadas (5a-5e).
        """
        # La técnica de albumentations que soporta YOLO se llama BoundingBoxFormat.YOLO
        return A.Compose([
            # 5a) Rotación (±15-30 grados)
            A.Rotate(limit=(-30, 30), p=0.8, border_mode=cv2.BORDER_CONSTANT),

            # 5f) Desplazamiento horizontal/vertical (usando Affine)
            A.Affine(shear=None, scale=None, translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)}, p=0.8, mode=cv2.BORDER_CONSTANT),

            # 5b) Escalado (0.8-1.2x)
            # Nota: Scale lo hace Affine, pero usaremos RandomScaleAndCrop para más variación y zoom (5e)
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.7), # Escala entre 0.8x y 1.2x

            # 5e) Zoom (crop y resize) - Se logra con RandomSizedCrop o combinado con RandomScale
            A.RandomSizedCrop(min_max_height=(int(700), int(1024)), height=1024, width=1024, p=0.5, interpolation=cv2.INTER_LINEAR),
            
            # 5c) Cambios de brillo y contraste
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            
            # 5d) Cambios de saturación (y otros colores)
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.7),
            
        ], 
        # Formato de las Bounding Boxes (Cajas Delimitadoras) para YOLO: [class_id, x_center, y_center, width, height]
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


    def _load_yolo_data(self, image_path):
        """
        Carga la imagen y las bounding boxes (etiquetas YOLO) asociadas.
        """
        # 1. Cargar la imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations usa RGB

        # 2. Cargar las etiquetas YOLO
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.original_label_dir, base_name + '.txt')

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Coordenadas YOLO normalizadas: x_c, y_c, w, h
                        coords = [float(p) for p in parts[1:]] 
                        bboxes.append(coords)
                        class_labels.append(class_id)

        return image, bboxes, class_labels

    def _save_augmented_data(self, image, bboxes, class_labels, original_filename, augmentation_index):
        """
        Guarda la imagen y las etiquetas transformadas.
        """
        base_name = os.path.splitext(original_filename)[0]
        
        # Nombre del nuevo archivo: billete1_10000_d_aug_1.jpg
        new_filename = f"{base_name}_aug_{augmentation_index}.jpg"
        new_label_filename = f"{base_name}_aug_{augmentation_index}.txt"

        # Guardar la imagen
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Volver a BGR para cv2.imwrite
        cv2.imwrite(os.path.join(self.augmented_img_dir, new_filename), image_bgr)

        # Guardar las etiquetas YOLO
        with open(os.path.join(self.augmented_label_dir, new_label_filename), 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                # bbox es [x_c, y_c, w, h] normalizado, listo para guardar
                line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                f.write(line)

    def augment_dataset(self, augmentations_per_image=5):
        """
        Procesa todas las imágenes originales y aplica el aumento.
        """
        for image_path in self.image_files:
            original_filename = os.path.basename(image_path)
            
            # Cargar imagen y etiquetas
            image, bboxes, class_labels = self._load_yolo_data(image_path)
            
            # Generar N versiones aumentadas
            for i in range(1, augmentations_per_image + 1):
                try:
                    # Aplicar la transformación
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    
                    # Guardar el resultado. Albumentations maneja automáticamente los bboxes transformados
                    self._save_augmented_data(
                        transformed['image'], 
                        transformed['bboxes'], 
                        transformed['class_labels'], 
                        original_filename, 
                        i
                    )
                except Exception as e:
                    print(f"Error al aumentar {original_filename} - Iteración {i}: {e}")
            
        print("Data Augmentation completado. Los nuevos datos están en la carpeta 'augmented'.")