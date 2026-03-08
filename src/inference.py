"""
Pipeline de inferencia para el Detector de Intrusiones.

Permite cargar un modelo entrenado y clasificar tráfico de red nuevo.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

from src.model_persistence import load_model

logger = logging.getLogger(__name__)


class IntrusionDetector:
    """
    Detector de intrusiones que utiliza un modelo previamente entrenado
    para clasificar tráfico de red.

    Uso:
        detector = IntrusionDetector('data/models/best_model.joblib')
        predicciones = detector.predict(X_nuevo)
        probabilidades = detector.predict_proba(X_nuevo)
    """

    def __init__(self, model_path):
        """
        Inicializa el detector cargando el modelo desde disco.

        Parámetros:
            model_path (str): Ruta al archivo del modelo serializado.

        Raises:
            ModelNotFoundError: Si el archivo del modelo no existe.
        """
        self.model, self.metadata = load_model(model_path)

    def predict(self, X):
        """
        Clasifica datos de tráfico de red.

        Parámetros:
            X: Datos a clasificar (DataFrame, ndarray o ruta a CSV).
                Debe tener las mismas características que los datos
                de entrenamiento (ya preprocesados y escalados).

        Retorna:
            np.ndarray: Array con las predicciones de clase.
        """
        X_input = self._prepare_input(X)
        return self.model.predict(X_input)

    def predict_proba(self, X):
        """
        Calcula las probabilidades de pertenencia a cada clase.

        Parámetros:
            X: Datos a clasificar (DataFrame, ndarray o ruta a CSV).

        Retorna:
            np.ndarray: Matriz de probabilidades (n_samples x n_classes).
        """
        X_input = self._prepare_input(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_input)
        raise AttributeError(
            "El modelo cargado no soporta predict_proba()."
        )

    def get_classes(self):
        """
        Obtiene las clases que el modelo puede predecir.

        Retorna:
            np.ndarray: Array con los nombres de las clases.
        """
        if hasattr(self.model, 'classes_'):
            return self.model.classes_
        raise AttributeError(
            "El modelo cargado no tiene el atributo classes_."
        )

    def _prepare_input(self, X):
        """
        Prepara los datos de entrada para la predicción.

        Parámetros:
            X: Datos de entrada (DataFrame, ndarray, o ruta a CSV).

        Retorna:
            np.ndarray o pd.DataFrame: Datos listos para predicción.

        Raises:
            InvalidDataError: Si el tipo de datos no es soportado.
            FileNotFoundError: Si la ruta a CSV no existe.
        """
        if isinstance(X, str):
            if not os.path.exists(X):
                raise FileNotFoundError(f"Archivo no encontrado: {X}")
            return pd.read_csv(X)
        elif isinstance(X, (pd.DataFrame, np.ndarray)):
            return X
        else:
            raise TypeError(
                f"Tipo de datos no soportado: {type(X)}. "
                "Se acepta DataFrame, ndarray o ruta a CSV."
            )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python -m src.inference <ruta_modelo> <ruta_datos_csv>")
        print("Ejemplo: python -m src.inference data/models/best_model.joblib datos_nuevos.csv")
        sys.exit(1)

    # Configurar logging para ejecución CLI
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    logger.info("Cargando modelo desde: %s", model_path)
    detector = IntrusionDetector(model_path)

    logger.info("Cargando datos desde: %s", data_path)
    datos = pd.read_csv(data_path)
    logger.info("Datos cargados: %d registros, %d características",
                datos.shape[0], datos.shape[1])

    logger.info("Realizando predicciones...")
    predicciones = detector.predict(datos)

    # Mostrar resumen de predicciones
    clases_unicas, conteos = np.unique(predicciones, return_counts=True)
    logger.info("--- Resumen de Predicciones ---")
    for clase, conteo in zip(clases_unicas, conteos):
        logger.info("  %s: %d (%.1f%%)", clase, conteo, conteo / len(predicciones) * 100)

    # Guardar resultados
    output_path = data_path.replace('.csv', '_predicciones.csv')
    resultado = datos.copy()
    resultado['Prediccion'] = predicciones
    resultado.to_csv(output_path, index=False)
    logger.info("Predicciones guardadas en: %s", output_path)
