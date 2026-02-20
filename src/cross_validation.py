"""
Módulo de validación cruzada para el Detector de Intrusiones.

Implementa StratifiedKFold para tener mayor confianza en los resultados
de los modelos, especialmente para clases minoritarias.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


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

    print(f"Iniciando validación cruzada con {n_splits} folds...")
    print(f"Tamaño del dataset: {X.shape[0]} muestras, {X.shape[1]} características")

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
    print("\n--- Resultados de Validación Cruzada ---")
    print(f"Accuracy:  {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
    print(f"Precision: {results['precision_mean']:.4f} (+/- {results['precision_std']:.4f})")
    print(f"Recall:    {results['recall_mean']:.4f} (+/- {results['recall_std']:.4f})")
    print(f"F1-Score:  {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")

    print("\n--- Scores por Fold ---")
    for i in range(n_splits):
        print(
            f"  Fold {i + 1}: "
            f"Acc={cv_results['test_accuracy'][i]:.4f}, "
            f"Prec={cv_results['test_precision_macro'][i]:.4f}, "
            f"Rec={cv_results['test_recall_macro'][i]:.4f}, "
            f"F1={cv_results['test_f1_macro'][i]:.4f}"
        )

    return results
