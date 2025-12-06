# -*- coding: utf-8 -*-
# src/main.py

import argparse
import os
import sys

# Añadir el directorio raíz del proyecto al PATH para imports relativos
# Esto asegura que los imports (ej: from src.core...) funcionen correctamente
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Agrega la raíz del proyecto (un nivel superior al directorio 'src')
# Esto permite que 'from src.core...' funcione.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT) 

from src.core.config_manager import ConfigManager
from src.core.dataset_manager import DatasetManager
from src.detection.yolov8_trainer import YOLOv8Trainer
from src.detection.real_time_detector import RealTimeDetector

def setup_project():
    """Configura los managers y genera el archivo dataset.yaml."""
    try:
        cfg_manager = ConfigManager()
        ds_manager = DatasetManager(cfg_manager)
        ds_manager.create_yolo_config()
        print(f"\n✅ Setup completado. Configuración cargada: {cfg_manager}")
        print(f"   Dataset Manager listo: {ds_manager}")
        print("\nAhora puede ejecutar el modo 'train'.")
        return cfg_manager, ds_manager
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error durante el setup: {e}")
        print("Asegúrese de que 'config/config.yaml' y 'data/classes.txt' existan y estén llenos.")
        return None, None

def main():
    """
    Punto de entrada principal para orquestar las operaciones del proyecto.
    """
    parser = argparse.ArgumentParser(description="Detector de Monedas y Billetes Colombianos con YOLOv8.")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'detect_rt', 'detect_img', 'setup'], 
                        help="Modo de operación: 'setup', 'train', 'detect_rt' (tiempo real) o 'detect_img' (imagen estática).")
    parser.add_argument('--image_path', type=str, default=None, 
                        help="Ruta de la imagen a detectar (solo para --mode detect_img).")
    args = parser.parse_args()

    # 1. Cargar la configuración y Managers
    if args.mode != 'setup':
        try:
            cfg_manager = ConfigManager()
            ds_manager = DatasetManager(cfg_manager)
        except Exception as e:
            print(f"\n❌ Error al cargar configuración. Intente ejecutar el modo 'setup' primero: {e}")
            return

    # 2. Lógica de Operación
    if args.mode == 'setup':
        setup_project()
    
    elif args.mode == 'train':
        trainer = YOLOv8Trainer(cfg_manager, ds_manager)
        trainer.train()

    elif args.mode == 'detect_rt':
        detector = RealTimeDetector(cfg_manager)
        detector.detect_real_time()

    elif args.mode == 'detect_img':
        if args.image_path is None:
            print("❌ Debe proporcionar la ruta de la imagen con --image_path.")
            return
        detector = RealTimeDetector(cfg_manager)
        detector.detect_static_image(args.image_path)
    
if __name__ == "__main__":
    main()

# --- FIN src/main.py ---