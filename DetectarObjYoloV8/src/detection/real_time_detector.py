# -*- coding: utf-8 -*-
# src/detection/real_time_detector.py
import cv2
import os
from ultralytics import YOLO
from src.core.config_manager import ConfigManager

class RealTimeDetector:
    """
    Clase para la detección de objetos en tiempo real desde cámara web o imágenes estáticas.
    """
    def __init__(self, config_manager: ConfigManager):
        self.cfg_yolo = config_manager.get('yolov8', None)
        self.cfg_rt = config_manager.get('realtime', None)
        self.paths = config_manager.get('paths', None)

        # 1. Cargar el mejor modelo entrenado
        #weights_dir = os.path.join(self.paths.get('weights_output'), 'colombian_currency_detector', 'weights')
        #weights_path = os.path.join(weights_dir, 'best.pt')
        weights_path = self.cfg_rt.get('model_path') # Lee la ruta exacta del config.yaml

        if not os.path.exists(weights_path):
            print(f"⚠️ Advertencia: Pesos entrenados no encontrados en {weights_path}. Usando modelo base...")
            model_size = self.cfg_yolo.get('model_size', 'n')
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
             self.model = YOLO(weights_path)
             print(f"✨ Pesos personalizados cargados exitosamente desde {weights_path}")
        
        self.imgsz = self.cfg_yolo.get('imgsz', 640)
        self.conf_thresh = self.cfg_yolo.get('confidence_threshold', 0.50)
        self.webcam_id = self.cfg_rt.get('webcam_id', 0)

    def detect_static_image(self, image_path: str):
        """
        Realiza la detección en una imagen estática y muestra el resultado.
        """
        if not os.path.exists(image_path):
            print(f"❌ Error: Imagen no encontrada en {image_path}")
            return
            
        print(f"▶️ Iniciando detección en imagen: {image_path}")
        
        # Ejecuta la inferencia
        results = self.model(image_path, 
                             imgsz=self.imgsz, 
                             conf=self.conf_thresh,
                             save=False # No guardar en disco
                            )
        
        # Muestra la imagen con los bounding boxes
        result_image = results[0].plot()
        cv2.imshow("Deteccion de Imagen Estatica", result_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        print("✅ Detección en imagen estática finalizada.")

    def detect_real_time(self):
        """
        Realiza la detección en tiempo real desde la cámara web.
        """
        cap = cv2.VideoCapture(self.webcam_id)
        if not cap.isOpened():
            print(f"❌ Error: No se puede acceder a la cámara con ID {self.webcam_id}. Verifique la conexión.")
            return

        print("▶️ Iniciando detección en tiempo real. Presione 'q' para salir...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: No se pudo capturar el frame.")
                break

            # InferenciA en el frame (stream=True optimiza para video)
            results = self.model(frame, 
                                 imgsz=self.imgsz, 
                                 conf=self.conf_thresh, 
                                 stream=True)

            # Procesa resultados. r.plot() dibuja BBs, etiquetas y confianza.
            annotated_frame = frame
            for r in results:
                annotated_frame = r.plot() 
                
            # Muestra el frame anotado
            cv2.imshow("Detector de Monedas Colombianas (YOLOv8)", annotated_frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(self.cfg_rt.get('delay_ms', 1)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detección en tiempo real finalizada.")

# --- FIN src/detection/real_time_detector.py ---