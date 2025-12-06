# Archivo README.md
# 💰 PONAO-IA: Detección de Monedas y Billetes Colombianos (YOLOv8)

Este proyecto implementa un sistema de Detección de Objetos en tiempo real utilizando la arquitectura YOLOv8, enfocado en la identificación de billetes y monedas de la denominación colombiana a través de cámara web e imágenes estáticas.

## ⚙️ Configuración del Entorno

1.  **Clonar el Repositorio** (Si no lo ha hecho):
    ```bash
    git clone [https://github.com/tu-usuario/PONAO-IA.git](https://github.com/tu-usuario/PONAO-IA.git)
    cd PONAO-IA
    git checkout DetectarObjYoloV8
    ```
2.  **Crear y Activar Entorno Virtual (Python 3.11)**:
    ```bash
    py -3.11 -m venv venv_ponao
    .\venv_ponao\Scripts\activate
    ```
3.  **Instalar Dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Uso del Proyecto

### 1. Preparación del Dataset

Coloque sus imágenes y archivos de etiquetas de LabelImg (.txt) en las carpetas `data/images/train`, `data/images/val`, `data/labels/train`, `data/labels/val`.

### 2. Configuración Inicial (Obligatorio)

Este paso genera el archivo YAML necesario para YOLOv8.

```bash
python src/main.py --mode setup

###  3. Entrenamiento del Modelo
```bash 
python src/main.py --mode train
### 4. Detección en Tiempo Real
```bash
python src/main.py --mode detect_rt
### 5. Detección en Imagen Estática
```bash
python src/main.py --mode detect_img --image_path 'ruta/a/mi/imagen_prueba.jpg'