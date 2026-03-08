"""
Módulo de detección de drift para el Detector de Intrusiones.

En un sistema de detección de intrusiones real, los patrones de ataque
evolucionan con el tiempo. Este módulo proporciona herramientas para
detectar cuándo el modelo puede estar degradándose:

- Data drift: cambios en la distribución de las features de entrada.
- Prediction drift: cambios en la distribución de las predicciones.
- DriftMonitor: clase para monitoreo continuo por batches.
"""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def detect_data_drift(X_reference, X_current, feature_names=None,
                      threshold=0.05, method='ks'):
    """
    Detecta drift en los datos comparando distribuciones de features
    entre un conjunto de referencia y datos actuales.

    Usa el test de Kolmogorov-Smirnov (KS) para comparar distribuciones
    continuas feature por feature.

    Parámetros:
        X_reference (np.ndarray): Datos de referencia (e.g., datos de entrenamiento).
        X_current (np.ndarray): Datos actuales a comparar.
        feature_names (list, opcional): Nombres de las features para el reporte.
        threshold (float): Nivel de significancia para rechazar la hipótesis nula
            de que las distribuciones son iguales. Default: 0.05.
        method (str): Método estadístico: 'ks' (Kolmogorov-Smirnov).

    Retorna:
        dict: Diccionario con:
            - 'has_drift': bool indicando si se detectó drift en al menos una feature.
            - 'drifted_features': Lista de features con drift.
            - 'n_drifted': Número de features con drift.
            - 'feature_results': Lista de dicts con detalles por feature.
            - 'drift_ratio': Proporción de features con drift.
    """
    if X_reference.ndim == 1:
        X_reference = X_reference.reshape(-1, 1)
    if X_current.ndim == 1:
        X_current = X_current.reshape(-1, 1)

    n_features = X_reference.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    logger.info("Detectando data drift (%s test, threshold=%.4f)...",
                method.upper(), threshold)
    logger.info("  Referencia: %d muestras, Actual: %d muestras",
                X_reference.shape[0], X_current.shape[0])

    feature_results = []
    drifted_features = []

    for i in range(n_features):
        if method == 'ks':
            stat, p_value = stats.ks_2samp(X_reference[:, i], X_current[:, i])
        else:
            raise ValueError(f"Método no soportado: {method}")

        is_drifted = p_value < threshold
        feature_results.append({
            'feature': feature_names[i],
            'statistic': stat,
            'p_value': p_value,
            'is_drifted': is_drifted,
        })

        if is_drifted:
            drifted_features.append(feature_names[i])

    n_drifted = len(drifted_features)
    drift_ratio = n_drifted / n_features if n_features > 0 else 0
    has_drift = n_drifted > 0

    if has_drift:
        logger.warning("⚠  Data drift detectado en %d/%d features (%.1f%%)",
                        n_drifted, n_features, drift_ratio * 100)
        logger.warning("  Features afectadas: %s", drifted_features[:10])
    else:
        logger.info("✓  No se detectó data drift significativo.")

    return {
        'has_drift': has_drift,
        'drifted_features': drifted_features,
        'n_drifted': n_drifted,
        'feature_results': feature_results,
        'drift_ratio': drift_ratio,
    }


def detect_prediction_drift(y_reference, y_current, threshold=0.05):
    """
    Detecta drift en las predicciones comparando distribuciones de clases
    entre un conjunto de referencia y predicciones actuales.

    Usa el test Chi-cuadrado para comparar frecuencias de clases.

    Parámetros:
        y_reference (np.ndarray): Predicciones/etiquetas de referencia.
        y_current (np.ndarray): Predicciones/etiquetas actuales.
        threshold (float): Nivel de significancia. Default: 0.05.

    Retorna:
        dict: Diccionario con:
            - 'has_drift': bool indicando si se detectó drift.
            - 'statistic': Estadístico chi-cuadrado.
            - 'p_value': Valor p.
            - 'reference_distribution': Distribución de referencia.
            - 'current_distribution': Distribución actual.
    """
    logger.info("Detectando prediction drift (chi-cuadrado, threshold=%.4f)...", threshold)

    # Obtener todas las clases únicas de ambos conjuntos
    all_classes = np.union1d(np.unique(y_reference), np.unique(y_current))

    # Contar frecuencias para cada clase
    ref_counts = np.array([np.sum(y_reference == c) for c in all_classes], dtype=float)
    cur_counts = np.array([np.sum(y_current == c) for c in all_classes], dtype=float)

    # Normalizar a proporciones
    ref_props = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts
    cur_props = cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else cur_counts

    # Calcular frecuencias esperadas (bajo la distribución de referencia)
    expected = ref_props * cur_counts.sum()
    # Evitar divisiones por cero
    expected = np.where(expected == 0, 1e-10, expected)

    stat, p_value = stats.chisquare(cur_counts, f_exp=expected)
    has_drift = p_value < threshold

    ref_dist = dict(zip(all_classes, ref_props))
    cur_dist = dict(zip(all_classes, cur_props))

    if has_drift:
        logger.warning("⚠  Prediction drift detectado (p-value=%.6f)", p_value)
    else:
        logger.info("✓  No se detectó prediction drift (p-value=%.6f)", p_value)

    return {
        'has_drift': has_drift,
        'statistic': stat,
        'p_value': p_value,
        'reference_distribution': ref_dist,
        'current_distribution': cur_dist,
    }


class DriftMonitor:
    """
    Monitor continuo de drift para el Detector de Intrusiones.

    Mantiene datos de referencia y permite evaluar batches entrantes
    para detectar data drift y prediction drift.

    Uso:
        monitor = DriftMonitor(X_train, y_train_pred)
        for batch_X, batch_y_pred in incoming_batches:
            report = monitor.check(batch_X, batch_y_pred)
            if report['data_drift']['has_drift']:
                alert("¡Reentrenamiento necesario!")
    """

    def __init__(self, X_reference, y_reference, feature_names=None,
                 data_threshold=0.05, prediction_threshold=0.05):
        """
        Inicializa el monitor con datos de referencia.

        Parámetros:
            X_reference (np.ndarray): Features de referencia.
            y_reference (np.ndarray): Predicciones/etiquetas de referencia.
            feature_names (list, opcional): Nombres de las features.
            data_threshold (float): Umbral para data drift.
            prediction_threshold (float): Umbral para prediction drift.
        """
        self.X_reference = np.asarray(X_reference)
        self.y_reference = np.asarray(y_reference)
        self.feature_names = feature_names
        self.data_threshold = data_threshold
        self.prediction_threshold = prediction_threshold
        self.history = []

        logger.info("DriftMonitor inicializado. Referencia: %d muestras, %d features",
                     self.X_reference.shape[0], self.X_reference.shape[1])

    def check(self, X_current, y_current=None):
        """
        Evalúa un batch de datos buscando drift.

        Parámetros:
            X_current (np.ndarray): Features del batch actual.
            y_current (np.ndarray, opcional): Predicciones del batch actual.

        Retorna:
            dict: Reporte con 'data_drift', 'prediction_drift' (si aplica),
                  y 'batch_number'.
        """
        X_current = np.asarray(X_current)

        report = {
            'batch_number': len(self.history) + 1,
            'data_drift': detect_data_drift(
                self.X_reference, X_current,
                feature_names=self.feature_names,
                threshold=self.data_threshold,
            ),
        }

        if y_current is not None:
            y_current = np.asarray(y_current)
            report['prediction_drift'] = detect_prediction_drift(
                self.y_reference, y_current,
                threshold=self.prediction_threshold,
            )

        self.history.append(report)

        needs_retrain = report['data_drift']['has_drift']
        if 'prediction_drift' in report:
            needs_retrain = needs_retrain or report['prediction_drift']['has_drift']

        report['needs_retraining'] = needs_retrain

        if needs_retrain:
            logger.warning("⚠  Batch %d: se recomienda reentrenamiento del modelo.",
                           report['batch_number'])

        return report
