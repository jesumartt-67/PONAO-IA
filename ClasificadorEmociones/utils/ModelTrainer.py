from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

class ModelTrainer:
    """
    Clase para entrenar el modelo, implementar callbacks y usar data augmentation.
    """
    def __init__(self, model, model_name='emotion_cnn.h5'):
        """
        Inicializa el ModelTrainer.

        :param model: El modelo Keras compilado a entrenar.
        :param model_name: Nombre para guardar el modelo entrenado.
        """
        self.model = model
        self.model_path = f'models/{model_name}'
        self.callbacks = self._get_callbacks()

    def _get_callbacks(self):
        """
        Configura los callbacks para el entrenamiento (Checkpointing, Early Stopping, LR Scheduler).
        
        :return: Lista de callbacks.
        """
        # Guardar el mejor modelo basado en la precisión de validación
        checkpoint = ModelCheckpoint(
            self.model_path, 
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True, 
            mode='max'
        )
        
        # Detener el entrenamiento si la precisión no mejora después de 15 épocas
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            verbose=1, 
            mode='min', 
            restore_best_weights=True
        )
        
        # Reducir la tasa de aprendizaje si la pérdida de validación se estanca
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=6, 
            verbose=1, 
            min_delta=0.0001
        )
        
        return [checkpoint, early_stop, reduce_lr]

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Entrena el modelo usando Data Augmentation.

        :param X_train: Características de entrenamiento.
        :param y_train: Etiquetas de entrenamiento.
        :param X_val: Características de validación.
        :param y_val: Etiquetas de validación.
        :param epochs: Número de épocas.
        :param batch_size: Tamaño del lote.
        :return: Objeto History de Keras.
        """
        print("Iniciando Data Augmentation...")
        # Generador de imágenes con aumento de datos para evitar overfitting
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(X_train)
        
        print("Iniciando entrenamiento del modelo...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=self.callbacks
        )
        print("Entrenamiento finalizado. El mejor modelo se guardó en:", self.model_path)
        return history