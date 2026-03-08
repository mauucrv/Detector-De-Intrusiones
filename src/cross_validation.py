"""
Módulo de validación cruzada para el Detector de Intrusiones.

Implementa StratifiedKFold para tener mayor confianza en los resultados
de los modelos, especialmente para clases minoritarias.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


def cross_validate_model(model, X, y, n_splits=5, random_state=42, n_jobs=-1):
    """
    Realiza validación cruzada estratificada de un modelo de clasificación.

    Usa StratifiedKFold para preservar la proporción de clases en cada fold,
    lo cual es crucial para datasets desbalanceados como CIC-IDS-2017.

    Parámetros:
        model: Modelo o pipeline de clasificación (con fit/predict).
        X: Features (ndarray o DataFrame).
        y: Target (ndarray o Series).
        n_splits (int): Número de folds para la validación cruzada.
        random_state (int): Semilla de reproducibilidad.
        n_jobs (int): Número de trabajos paralelos (-1 para usar todos los cores).

    Retorna:
        dict: Diccionario con las estadísticas de validación cruzada:
            - 'accuracy_mean', 'accuracy_std': Media y desviación estándar de accuracy.
            - 'precision_mean', 'precision_std': Media y desviación estándar de precision (macro).
            - 'recall_mean', 'recall_std': Media y desviación estándar de recall (macro).
            - 'f1_mean', 'f1_std': Media y desviación estándar de f1-score (macro).
            - 'scores_per_fold': Diccionario con los scores individuales por fold.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Definir las métricas a evaluar
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(
            precision_score, average='macro', zero_division=0
        ),
        'recall_macro': make_scorer(
            recall_score, average='macro', zero_division=0
        ),
        'f1_macro': make_scorer(
            f1_score, average='macro', zero_division=0
        ),
    }

    logger.info("Iniciando validación cruzada con %d folds...", n_splits)
    logger.info("Tamaño del dataset: %d muestras, %d características",
                X.shape[0], X.shape[1])

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    # Extraer y reportar resultados
    results = {
        'accuracy_mean': np.mean(cv_results['test_accuracy']),
        'accuracy_std': np.std(cv_results['test_accuracy']),
        'precision_mean': np.mean(cv_results['test_precision_macro']),
        'precision_std': np.std(cv_results['test_precision_macro']),
        'recall_mean': np.mean(cv_results['test_recall_macro']),
        'recall_std': np.std(cv_results['test_recall_macro']),
        'f1_mean': np.mean(cv_results['test_f1_macro']),
        'f1_std': np.std(cv_results['test_f1_macro']),
        'scores_per_fold': {
            'accuracy': cv_results['test_accuracy'].tolist(),
            'precision_macro': cv_results['test_precision_macro'].tolist(),
            'recall_macro': cv_results['test_recall_macro'].tolist(),
            'f1_macro': cv_results['test_f1_macro'].tolist(),
        },
    }

    # Imprimir resumen
    logger.info("--- Resultados de Validación Cruzada ---")
    logger.info("Accuracy:  %.4f (+/- %.4f)", results['accuracy_mean'], results['accuracy_std'])
    logger.info("Precision: %.4f (+/- %.4f)", results['precision_mean'], results['precision_std'])
    logger.info("Recall:    %.4f (+/- %.4f)", results['recall_mean'], results['recall_std'])
    logger.info("F1-Score:  %.4f (+/- %.4f)", results['f1_mean'], results['f1_std'])

    logger.debug("--- Scores por Fold ---")
    for i in range(n_splits):
        logger.debug(
            "  Fold %d: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
            i + 1,
            cv_results['test_accuracy'][i],
            cv_results['test_precision_macro'][i],
            cv_results['test_recall_macro'][i],
            cv_results['test_f1_macro'][i],
        )

    return results
