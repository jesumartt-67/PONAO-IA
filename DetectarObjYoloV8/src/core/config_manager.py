# -*- coding: utf-8 -*-
# src/core/config_manager.py
import yaml
import os

class ConfigManager:
    """
    Clase para gestionar la configuración centralizada del proyecto
    desde el archivo YAML (config/config.yaml).
    """
    def __init__(self, config_path=os.path.join('config', 'config.yaml')):
        # La ruta se resuelve relativa a la raíz del proyecto (DetectarObjYoloV8)
        # 1. Obtiene la ruta del directorio actual (src/core/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. Sube dos niveles (a DetectarObjYoloV8/) y luego va a 'config/config.yaml'
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # 3. CONSTRUYE la ruta absoluta al archivo config.yaml:
        self.config_path = os.path.join(project_root, config_path)
        # --- FIN CÓDIGO MODIFICADO ---
        #self.config_path = os.path.join(os.getcwd(), config_path)
        self.config = self._load_config()

    def _load_config(self):
        """Carga el archivo YAML."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"❌ Error: Archivo de configuración no encontrado en: {self.config_path}"
            )
        try:
            # AÑADIMOS encoding='utf-8' AQUÍ para forzar la lectura correcta
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"❌ Error al parsear el archivo YAML: {e}")
        except UnicodeDecodeError as e:
            # Capturamos el error de codificación si falla incluso con UTF-8
            raise ValueError(f"❌ Error crítico de codificación en config.yaml: {e}")

    def get(self, section, key=None, default=None):
        """
        Obtiene un valor de la configuración por sección y clave.
        Si solo se proporciona la sección, devuelve todo el diccionario de esa sección.
        """
        section_data = self.config.get(section)
        if section_data is None:
            return default
        
        if key is None:
            return section_data
        
        return section_data.get(key, default)

    def __str__(self):
        """Representación en string de la configuración cargada."""
        return f"Configuración cargada (v{self.get('general', 'version')}): {self.config_path}"

# --- FIN src/core/config_manager.py ---