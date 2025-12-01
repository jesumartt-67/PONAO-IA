import sys
import os
# Ajusta el path para importar las clases del paquete utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from DataLoader import DataLoader # type: ignore
from EmotionModel import EmotionModel # type: ignore
from ModelTrainer import ModelTrainer # type: ignore
from RealTimeDetector import RealTimeDetector # type: ignore

# --- Configuración General ---
DATASET_PATH = 'data/fer2013.csv'
MODEL_FILENAME = 'emotion_cnn.h5'

def train_workflow():
    """Flujo de trabajo completo para cargar datos, construir y entrenar el modelo."""
    print("--- Iniciando Flujo de Entrenamiento ---")
    
    # 1. Cargar y Preprocesar Datos
    data_loader = DataLoader(data_path=DATASET_PATH)
    try:
        X_train, y_train, X_val, y_val, _, _ = data_loader.load_and_preprocess()
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo FER2013 en {DATASET_PATH}.")
        print("Por favor, descarga y coloca el archivo 'fer2013.csv' en la carpeta 'data/'.")
        return

    # 2. Construir el Modelo CNN
    emotion_model_builder = EmotionModel(num_classes=data_loader.num_classes)
    model = emotion_model_builder.build_model()
    # model.summary() # Descomentar para ver la arquitectura

    # 3. Entrenar el Modelo
    trainer = ModelTrainer(model=model, model_name=MODEL_FILENAME)
    # Se recomiendan más épocas (50+) pero se usa un número bajo para una prueba rápida
    trainer.train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=64) 
    
    print("--- Entrenamiento Finalizado Exitosamente ---")

def detect_workflow():
    """Flujo de trabajo para la detección en tiempo real con el modelo guardado."""
    print("--- Iniciando Detección en Tiempo Real ---")
    
    # Instancia DataLoader solo para obtener las etiquetas, no para cargar el dataset completo
    data_loader_temp = DataLoader() 
    emotion_labels = data_loader_temp.get_labels()

    # 1. Inicializar y Ejecutar el Detector
    detector = RealTimeDetector(
        model_path=f'models/{MODEL_FILENAME}',
        emotion_labels=emotion_labels
    )
    detector.start_detection()
    
    print("--- Detección en Tiempo Real Finalizada ---")


if __name__ == "__main__":
    # La variable sys.argv[1] permite pasar 'train' o 'detect' como argumento de línea de comandos.
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'train':
        train_workflow()
    elif len(sys.argv) > 1 and sys.argv[1].lower() == 'detect':
        detect_workflow()
    else:
        print("Uso:")
        print("Para ENTRENAR el modelo: python main.py train")
        print("Para DETECTAR en tiempo real: python main.py detect")
        print("Asegúrate de tener un modelo entrenado antes de usar 'detect'.")