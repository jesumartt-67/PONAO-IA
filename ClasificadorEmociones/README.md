
# Documentación del proyecto
# visionA: Clasificador de Emociones Faciales en Tiempo Real 🤯

## 📄 Descripción

Este proyecto implementa un sistema de Machine Learning para la **clasificación de emociones faciales en tiempo real** utilizando el dataset **FER2013** y una Red Neuronal Convolucional (CNN) implementada en TensorFlow/Keras. Utiliza OpenCV para la captura de video en vivo y detección de rostros mediante Haar Cascades. El código está estructurado bajo el paradigma de **Programación Orientada a Objetos (POO)** en Python para modularidad.

## ⚙️ Requisitos

* **Sistema Operativo:** Windows 11
* **Python:** 3.11.0
* **TensorFlow >= 2.13 ya soporta oficialmente Python 3.11. 
* **Hardware:** Tarjeta gráfica compatible con CUDA (opcional, para entrenamiento rápido)

## 🛠️ Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/TuUsuario/visionA.git](https://github.com/TuUsuario/visionA.git)
    cd visionA
    ```

2.  **Crea y activa el entorno virtual:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga el Dataset:**
    Descarga el archivo `fer2013.csv` y colócalo en la carpeta **`data/`**.

## 📂 Estructura de Carpetas
visionArtificial/
├── data/              # Almacenará el dataset FER2013 y datos preprocesados
├── models/            # Modelos entrenados (.h5 o .tf) y checkpoints
├── scripts/
│   └── train.py       # Script principal de entrenamiento (puede ser movido a main.py)
├── utils/             # Clases con lógica de negocio (POO)
│   ├── DataLoader.py
│   ├── EmotionModel.py
│   ├── ModelTrainer.py
│   ├── RealTimeDetector.py
│   └── __init__.py    # Para que Python trate 'utils' como un paquete
├── main.py            # Orquestador del proyecto (detección en tiempo real o llamada a train.py)
├── requirements.txt   # Lista de dependencias del proyecto
└── README.md          # Documentación del proyecto

5. 🧠 Entrenamiento y 6. 📸 Detección
🚀 Uso del Proyecto
Entrenamiento del Modelo: Este comando cargará fer2013.csv, construirá el modelo, lo entrenará y guardará el mejor modelo como models/emotion_cnn.h5
=> python main.py train
Detección en Tiempo Real: Este comando abrirá tu cámara web y comenzará a clasificar las emociones de las caras detectadas en tiempo real. Presiona 'q' para salir.
# Asegúrate de que el modelo emotion_cnn.h5 existe
=> python main.py detect 

# jesus-ponao-ia
programacion IA 
