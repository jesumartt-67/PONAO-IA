import os
from src.augmentor import YoloAugmentor

# 1. Definición de Rutas (usando la estructura que creamos)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Rutas originales
ORIGINAL_IMG_DIR = os.path.join(DATA_DIR, 'original', 'images')
ORIGINAL_LABEL_DIR = os.path.join(DATA_DIR, 'original', 'labels')

# Rutas de salida (augmented)
AUGMENTED_IMG_DIR = os.path.join(DATA_DIR, 'augmented', 'images')
AUGMENTED_LABEL_DIR = os.path.join(DATA_DIR, 'augmented', 'labels')

# 2. Configuración de Aumento
# Si tienes 64 imágenes originales y quieres 5 aumentos por imagen:
# 64 originales + (64 * 5 aumentos) = 384 imágenes totales en tu nuevo dataset
AUGMENTS_PER_IMAGE = 5 

if __name__ == "__main__":
    # Asegurar que los directorios de salida existen
    for d in [AUGMENTED_IMG_DIR, AUGMENTED_LABEL_DIR]:
        os.makedirs(d, exist_ok=True)
        
    # Crear una instancia de la clase YoloAugmentor
    augmentor = YoloAugmentor(
        ORIGINAL_IMG_DIR,
        ORIGINAL_LABEL_DIR,
        AUGMENTED_IMG_DIR,
        AUGMENTED_LABEL_DIR
    )

    # Ejecutar el aumento de datos
    augmentor.augment_dataset(AUGMENTS_PER_IMAGE)