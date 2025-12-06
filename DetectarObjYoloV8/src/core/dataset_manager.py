# -*- coding: utf-8 -*-
# src/core/dataset_manager.py
import yaml
import os
from src.core.config_manager import ConfigManager # Importamos la clase anterior

class DatasetManager:
    """
    Gestiona la organización y el archivo de configuración del dataset
    en formato YOLOv8. Crea el dataset.yaml requerido para el entrenamiento.
    """
    def __init__(self, config_manager: ConfigManager):
        self.cfg = config_manager
        self.data_root = self.cfg.get('paths', 'data_root')
        self.data_config_path = self.cfg.get('paths', 'data_config')
        self.class_names_file = self.cfg.get('paths', 'class_names_file')
        self.names = self._load_class_names()
        
        # Validar si el directorio raíz de datos existe
        if not os.path.isdir(self.data_root):
            print(f"⚠️ Advertencia: El directorio {self.data_root} no existe. Créelo antes de entrenar.")

    def _load_class_names(self):
        """Carga los nombres de clases desde el archivo de texto consolidado."""
        full_path = self.class_names_file
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"❌ Error: Archivo de clases no encontrado en {full_path}. Cree la lista de denominaciones."
            )
            
        with open(full_path, 'r', encoding='utf-8') as f:
            # Lee, limpia espacios y líneas vacías
            class_list = [line.strip() for line in f if line.strip()]
            
        if not class_list:
            raise ValueError("❌ Error: El archivo de clases está vacío.")
            
        return class_list

    def create_yolo_config(self):
        """
        Crea o actualiza el archivo YAML requerido por YOLOv8 para el entrenamiento.
        """
        # La ruta 'path' debe ser relativa al archivo dataset.yaml, 
        # o absoluta, pero 'ultralytics' recomienda la ruta base y las relativas.
        # Aquí usamos la ruta relativa desde data/config hasta la raíz 'data/'
        
        config = {
            # 'path' apunta a la carpeta 'data'
            'path': os.path.join(os.getcwd(), self.data_root),
            
            # Rutas de subcarpetas relativas a 'path'
            'train': self.cfg.get('paths', 'images_train').replace(self.data_root + os.sep, ''),
            'val': self.cfg.get('paths', 'images_val').replace(self.data_root + os.sep, ''),
            
            'nc': len(self.names),
            # Usamos el formato de lista concisa
            'names': self.names 
        }

        # Asegura que la carpeta de configuración de datos exista
        os.makedirs(os.path.dirname(self.data_config_path), exist_ok=True)
        
        with open(self.data_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✅ Archivo de configuración YOLOv8 creado en: {self.data_config_path}")
        print(f"   Clases detectadas: {config['nc']}")

    def __str__(self):
        return (f"DatasetManager configurado para {len(self.names)} clases.\n"
                f"Ruta de configuración YOLO: {self.data_config_path}")

# --- FIN src/core/dataset_manager.py ---