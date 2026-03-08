"""
Módulo de optimización de hiperparámetros para el Detector de Intrusiones.

Implementa RandomizedSearchCV para buscar los mejores hiperparámetros
de cualquier estimator, con distribuciones predefinidas para
RandomForest y Gradient Boosting.
"""

import logging

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

logger = logging.getLogger(__name__)


def get_rf_param_distributions():
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


# Alias de compatibilidad con código anterior
get_param_distributions = get_rf_param_distributions


def get_gb_param_distributions():
    """
    Define el espacio de búsqueda de hiperparámetros para GradientBoostingClassifier
    o modelos compatibles (XGBoost, LightGBM con API de scikit-learn).

    Retorna:
        dict: Diccionario con distribuciones de hiperparámetros.
    """
    return {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    }


def tune_model(estimator, param_distributions, X_train, y_train,
               n_iter=20, cv=3, random_state=42, n_jobs=-1, scoring='f1_macro'):
    """
    Optimiza los hiperparámetros de cualquier estimator usando
    RandomizedSearchCV con validación cruzada estratificada.

    Parámetros:
        estimator: Modelo de clasificación (cualquier estimator de scikit-learn).
        param_distributions (dict): Espacio de búsqueda de hiperparámetros.
            Usar get_rf_param_distributions() o get_gb_param_distributions()
            como punto de partida.
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
            - 'train_test_gap': Diferencia entre score de train y validación.
    """
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    logger.info("Iniciando búsqueda de hiperparámetros...")
    logger.info("  - Estimator: %s", type(estimator).__name__)
    logger.info("  - Combinaciones a probar: %d", n_iter)
    logger.info("  - Folds de validación cruzada: %d", cv)
    logger.info("  - Métrica de optimización: %s", scoring)
    logger.debug("  - Espacio de búsqueda:")
    for param, values in param_distributions.items():
        logger.debug("    %s: %s", param, values)

    search = RandomizedSearchCV(
        estimator=estimator,
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
    logger.info("--- Resultados de la Optimización ---")
    logger.info("Mejor score (%s): %.4f", scoring, search.best_score_)
    logger.info("Mejores hiperparámetros:")
    for param, value in search.best_params_.items():
        logger.info("  %s: %s", param, value)

    # Detectar posible sobreajuste comparando train vs test score
    best_idx = search.best_index_
    train_score = search.cv_results_['mean_train_score'][best_idx]
    test_score = search.cv_results_['mean_test_score'][best_idx]
    gap = train_score - test_score

    logger.info("--- Análisis de Sobreajuste ---")
    logger.info("Score de entrenamiento: %.4f", train_score)
    logger.info("Score de validación:    %.4f", test_score)
    logger.info("Gap (train - test):     %.4f", gap)

    if gap > 0.05:
        logger.warning(
            "⚠  Gap significativo detectado (%.4f). "
            "El modelo podría estar sobreajustado. "
            "Considere incrementar min_samples_leaf o reducir max_depth.", gap
        )
    else:
        logger.info("✓  El gap train/test es aceptable.")

    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_,
        'train_test_gap': gap,
    }


def tune_random_forest(X_train, y_train, n_iter=20, cv=3, random_state=42,
                       n_jobs=-1, scoring='f1_macro'):
    """
    Wrapper de conveniencia: optimiza un RandomForestClassifier.

    Equivalente a llamar tune_model() con un RandomForestClassifier
    y get_rf_param_distributions().

    Parámetros:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        n_iter, cv, random_state, n_jobs, scoring: Ver tune_model().

    Retorna:
        dict: Ver tune_model().
    """
    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    param_distributions = get_rf_param_distributions()
    return tune_model(
        estimator, param_distributions, X_train, y_train,
        n_iter=n_iter, cv=cv, random_state=random_state,
        n_jobs=n_jobs, scoring=scoring,
    )
