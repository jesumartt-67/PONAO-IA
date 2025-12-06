# -*- coding: utf-8 -*-
# src/detection/yolov8_trainer.py
import os
from ultralytics import YOLO
from src.core.config_manager import ConfigManager
from src.core.dataset_manager import DatasetManager

class YOLOv8Trainer:
    """
    Clase para manejar el entrenamiento de un modelo YOLOv8.
    """
    def __init__(self, config_manager: ConfigManager, dataset_manager: DatasetManager):
        self.cfg = config_manager.get('yolov8', None)
        self.paths = config_manager.get('paths', None)
        self.ds_mgr = dataset_manager
        
        if not self.cfg or not self.paths:
            raise ValueError("❌ Configuración yolov8 o paths faltante en config.yaml.")

        model_size = self.cfg.get('model_size', 'n')
        
        # Carga el modelo base (e.g., yolov8s.pt)
        self.model = YOLO(f'yolov8{model_size}.pt') 
        print(f"✨ Modelo YOLOv8-{model_size} cargado exitosamente.")

    def train(self):
        """
        Inicia el proceso de entrenamiento de YOLOv8 utilizando los parámetros del config.yaml.
        """
        print("▶️ Iniciando entrenamiento de YOLOv8...")
        
        data_config_path = self.paths.get('data_config')
        
        if not os.path.exists(data_config_path):
             print(f"⚠️ Advertencia: El archivo {data_config_path} no existe. Ejecutando SETUP...")
             self.ds_mgr.create_yolo_config()
             
        # Parámetros pasados al método train de ultralytics
        results = self.model.train(
            data=data_config_path, 
            epochs=self.cfg.get('epochs'), 
            imgsz=self.cfg.get('imgsz'), 
            batch=self.cfg.get('batch'),
            lr0=self.cfg.get('lr0'),
            project=self.paths.get('weights_output'),
            name='colombian_currency_detector', # Nombre de la corrida 
            save=True,
            val=True
        )
        
        print("✅ Entrenamiento completado. Revise 'models/weights/colombian_currency_detector' para los resultados.")
        return results

# --- FIN src/detection/yolov8_trainer.py ---