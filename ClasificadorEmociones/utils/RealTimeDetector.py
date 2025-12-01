import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

class RealTimeDetector:
    """
    Clase para realizar la detección y clasificación de emociones en tiempo real 
    utilizando la cámara web y un modelo CNN pre-entrenado.
    """
    def __init__(self, model_path: str, emotion_labels: dict):
        """
        Inicializa el detector.

        :param model_path: Ruta al archivo del modelo Keras entrenado.
        :param emotion_labels: Diccionario de etiquetas {0: 'Angry', ...}.
        """
        self.model_path = model_path
        self.emotion_labels = emotion_labels
        self.model = self._load_model()
        # Inicializa el detector de caras (usando un clasificador Haar Cascade de OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _load_model(self):
        """Carga el modelo Keras entrenado desde el disco."""
        try:
            model = load_model(self.model_path)
            print(f"Modelo cargado exitosamente desde: {self.model_path}")
            return model
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Asegúrate de haber entrenado y guardado un modelo en la ruta especificada.")
            return None

    def start_detection(self):
        """
        Inicia la captura de video en vivo, detecta caras, clasifica la emoción 
        y muestra el resultado en la ventana de video.
        """
        if self.model is None:
            return

        cap = cv2.VideoCapture(0)  # 0 es la cámara web por defecto

        if not cap.isOpened():
            print("Error: No se puede abrir la cámara.")
            return

        print("Presiona 'q' para salir de la detección en tiempo real.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detectar caras en el frame en escala de grises
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extraer la Región de Interés (ROI) de la cara
                roi_gray = gray_frame[y:y+h, x:x+w]
                
                # Preprocesar la ROI para la entrada del modelo
                # 1. Redimensionar a 48x48
                # 2. Convertir a formato float32 y normalizar
                # 3. Remodelar para Keras (1, 48, 48, 1)
                
                # Asegura que la ROI no está vacía antes de redimensionar
                if roi_gray.shape[0] > 0 and roi_gray.shape[1] > 0:
                    try:
                        cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                        img_array = cropped_img.astype('float32') / 255.0
                        img_array = np.expand_dims(img_array, axis=-1)  # Agrega canal
                        img_array = np.expand_dims(img_array, axis=0)  # Agrega dimensión de lote
                        
                        # Predecir la emoción
                        predictions = self.model.predict(img_array, verbose=0)[0]
                        emotion_index = np.argmax(predictions)
                        emotion = self.emotion_labels[emotion_index]
                        confidence = predictions[emotion_index] * 100
                        
                        # Dibujar el rectángulo y el texto de la emoción
                        text = f"{emotion}: {confidence:.2f}%"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    except cv2.error as e:
                        # Manejo de error si el recorte o redimensionamiento falla
                        print(f"Error de procesamiento de imagen: {e}")
                        continue


            cv2.imshow('Emotion Detection', frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()