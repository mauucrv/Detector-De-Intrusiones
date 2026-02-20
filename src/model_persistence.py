"""
Funciones de persistencia de modelos.

Permite serializar y deserializar modelos entrenados con joblib,
incluyendo metadatos opcionales como nombre, métricas y fecha.
"""

import os
from datetime import datetime
from joblib import dump, load


def save_model(model, path, metadata=None):
    """
    Serializa un modelo entrenado junto con metadatos opcionales.

    Parámetros:
        model: Modelo o pipeline entrenado a serializar.
        path (str): Ruta del archivo donde se guardará el modelo.
        metadata (dict, opcional): Diccionario con metadatos adicionales
            (por ejemplo, nombre del modelo, métricas, fecha de entrenamiento).

    Retorna:
        str: Ruta absoluta del archivo guardado.
    """
    # Crear el directorio si no existe
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Construir el objeto a serializar
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
    }

    dump(model_data, path)
    print(f"Modelo guardado exitosamente en: {path}")
    return os.path.abspath(path)


def load_model(path):
    """
    Carga un modelo serializado desde disco.

    Parámetros:
        path (str): Ruta del archivo del modelo serializado.

    Retorna:
        tuple: (model, metadata) donde model es el modelo/pipeline
            entrenado y metadata es el diccionario de metadatos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no tiene el formato esperado.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {path}")

    model_data = load(path)

    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        metadata = model_data.get('metadata', {})
        metadata['saved_at'] = model_data.get('saved_at', 'desconocido')
        print(f"Modelo cargado exitosamente desde: {path}")
        print(f"Fecha de guardado: {metadata['saved_at']}")
        return model, metadata

    # Compatibilidad: si el archivo solo contiene el modelo directamente
    print(f"Modelo cargado exitosamente desde: {path} (formato legacy)")
    return model_data, {}
