"""
Módulo de optimización de hiperparámetros para el Detector de Intrusiones.

Implementa RandomizedSearchCV para buscar los mejores hiperparámetros
del RandomForestClassifier, incluyendo parámetros de regularización
para combatir el sobreajuste.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def get_param_distributions():
    """
    Define el espacio de búsqueda de hiperparámetros para RandomForestClassifier.

    Incluye parámetros de regularización para mitigar sobreajuste:
    - max_depth: limita la profundidad del árbol para evitar memorización.
    - min_samples_leaf: impone un mínimo de muestras por hoja.
    - min_samples_split: requiere más muestras para dividir un nodo.
    - max_features: limita las features consideradas en cada split.

    Retorna:
        dict: Diccionario con distribuciones de hiperparámetros.
    """
    return {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
    }


def tune_random_forest(X_train, y_train, n_iter=20, cv=3, random_state=42,
                        n_jobs=-1, scoring='f1_macro'):
    """
    Optimiza los hiperparámetros de un RandomForestClassifier usando
    RandomizedSearchCV con validación cruzada estratificada.

    La grilla de parámetros incluye controles de regularización para
    combatir el sobreajuste típico en datasets como CIC-IDS-2017.

    Parámetros:
        X_train: Features de entrenamiento (ndarray o DataFrame).
        y_train: Target de entrenamiento (ndarray o Series).
        n_iter (int): Número de combinaciones de parámetros a probar.
        cv (int): Número de folds para la validación cruzada interna.
        random_state (int): Semilla de reproducibilidad.
        n_jobs (int): Número de trabajos paralelos.
        scoring (str): Métrica a optimizar (por defecto 'f1_macro' para
            manejar bien el desbalance de clases).

    Retorna:
        dict: Diccionario con:
            - 'best_model': El mejor modelo encontrado (ya entrenado).
            - 'best_params': Los mejores hiperparámetros.
            - 'best_score': El mejor score obtenido.
            - 'cv_results': Resultados completos de la búsqueda.
    """
    base_model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    param_distributions = get_param_distributions()

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    print(f"Iniciando búsqueda de hiperparámetros...")
    print(f"  - Combinaciones a probar: {n_iter}")
    print(f"  - Folds de validación cruzada: {cv}")
    print(f"  - Métrica de optimización: {scoring}")
    print(f"  - Espacio de búsqueda:")
    for param, values in param_distributions.items():
        print(f"    {param}: {values}")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    # Reportar resultados
    print("\n--- Resultados de la Optimización ---")
    print(f"Mejor score ({scoring}): {search.best_score_:.4f}")
    print(f"Mejores hiperparámetros:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    # Detectar posible sobreajuste comparando train vs test score
    best_idx = search.best_index_
    train_score = search.cv_results_['mean_train_score'][best_idx]
    test_score = search.cv_results_['mean_test_score'][best_idx]
    gap = train_score - test_score

    print(f"\n--- Análisis de Sobreajuste ---")
    print(f"Score de entrenamiento: {train_score:.4f}")
    print(f"Score de validación:    {test_score:.4f}")
    print(f"Gap (train - test):     {gap:.4f}")

    if gap > 0.05:
        print("⚠  ADVERTENCIA: Gap significativo detectado. "
              "El modelo podría estar sobreajustado.")
        print("   Considere incrementar min_samples_leaf o reducir max_depth.")
    else:
        print("✓  El gap train/test es aceptable.")

    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_,
        'train_test_gap': gap,
    }
