import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore

class DataLoader:
    """
    Clase encargada de cargar el dataset FER2013 y realizar el preprocesamiento necesario.
    Convierte las cadenas de píxeles a arrays numpy y normaliza las imágenes.
    """
    def __init__(self, data_path: str = 'data/fer2013.csv'):
        """
        Inicializa el DataLoader con la ruta del archivo CSV de FER2013.

        :param data_path: Ruta al archivo fer2013.csv.
        """
        self.data_path = data_path
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.num_classes = len(self.emotion_labels)

    def load_and_preprocess(self):
        """
        Carga los datos, los divide en características (X) y etiquetas (y), 
        y los preprocesa para el entrenamiento de la CNN.
        
        Pasos de preprocesamiento:
        1. Parsear la cadena de píxeles a un array de 48x48.
        2. Normalizar los valores de píxeles a [0, 1].
        3. Remodelar (reshape) para el formato de entrada de Keras (48, 48, 1).
        4. Codificación one-hot de las etiquetas.

        :return: Tupla con (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        print(f"Cargando datos desde: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Función auxiliar para convertir cadena de píxeles a array 48x48
        def process_pixels(pixel_str):
            # Convierte la cadena de píxeles separada por espacios en un array numpy
            pixels = np.array(pixel_str.split(), 'float64')
            # Remodela el array a una matriz 48x48
            return pixels.reshape(48, 48)

        # Aplicar la función a la columna 'pixels'
        df['pixels'] = df['pixels'].apply(process_pixels)

        # Normalización y Remodelación (para CNN: altura, ancho, canales)
        X = np.stack(df['pixels'].values)
        X = X / 255.0  # Normalización
        X = X.reshape(-1, 48, 48, 1) # (N, 48, 48, 1)
        
        # One-hot Encoding para etiquetas
        y = to_categorical(df['emotion'], num_classes=self.num_classes)

        # División en conjuntos (Training, Validation, Test) según la columna 'Usage'
        X_train = X[df['Usage'] == 'Training']
        y_train = y[df['Usage'] == 'Training']
        
        X_val = X[df['Usage'] == 'PublicTest']
        y_val = y[df['Usage'] == 'PublicTest']

        X_test = X[df['Usage'] == 'PrivateTest']
        y_test = y[df['Usage'] == 'PrivateTest']

        print(f"Datos cargados. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_labels(self):
        """Retorna el diccionario de etiquetas de emoción."""
        return self.emotion_labels

# Nota: Asegúrate de colocar el archivo fer2013.csv en la carpeta data/