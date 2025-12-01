from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

class EmotionModel:
    """
    Clase para construir y compilar el modelo de Red Neuronal Convolucional (CNN).
    Utiliza una arquitectura VGG-like para la clasificación de emociones.
    """
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Inicializa el EmotionModel.

        :param input_shape: Forma de la imagen de entrada (altura, ancho, canales).
        :param num_classes: Número de clases de emociones.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Define y construye la arquitectura de la CNN.
        Se usa el regularizador L2 y BatchNormalization para mejorar la estabilidad y evitar el overfitting.
        
        :return: El modelo Keras construido y compilado.
        """
        model = Sequential()
        
        # Bloque 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Bloque 2
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Capa de Clasificación
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilación del modelo
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        self.model = model
        print("Modelo CNN construido y compilado.")
        return model

    def get_model(self):
        """Retorna el modelo Keras construido."""
        return self.model