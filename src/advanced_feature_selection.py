"""
Módulo de selección avanzada de features para el Detector de Intrusiones.

Complementa la selección basada en SelectFromModel del módulo feature_engineering
con técnicas más rigurosas:
- Recursive Feature Elimination (RFECV)
- Permutation Importance
- Comparación de múltiples métodos
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def recursive_feature_elimination(model, X, y, min_features=5, cv=3,
                                  scoring='f1_macro', n_jobs=-1, random_state=42):
    """
    Selección de features usando Recursive Feature Elimination con
    validación cruzada (RFECV).

    RFECV elimina features iterativamente, entrenando el modelo en cada paso,
    y selecciona el número óptimo de features según el score de CV.

    Parámetros:
        model: Estimator con atributo feature_importances_ o coef_.
        X: Features (DataFrame o ndarray).
        y: Target (Series o ndarray).
        min_features (int): Número mínimo de features a considerar.
        cv (int): Número de folds para la validación cruzada.
        scoring (str): Métrica de evaluación.
        n_jobs (int): Número de trabajos paralelos.
        random_state (int): Semilla de reproducibilidad.

    Retorna:
        dict: Diccionario con:
            - 'selected_features': Nombres/índices de features seleccionadas.
            - 'n_features_optimal': Número óptimo de features.
            - 'ranking': Ranking de importancia de cada feature.
            - 'cv_results': Scores de CV para cada número de features.
            - 'selector': El objeto RFECV ajustado.
    """
    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    logger.info("Iniciando Recursive Feature Elimination...")
    logger.info("  Features iniciales: %d", X.shape[1])
    logger.info("  Mínimo de features: %d", min_features)

    rfecv = RFECV(
        estimator=model,
        min_features_to_select=min_features,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    rfecv.fit(X, y)

    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[rfecv.support_].tolist()
    else:
        selected_features = np.where(rfecv.support_)[0].tolist()

    logger.info("Features seleccionadas por RFECV: %d de %d",
                rfecv.n_features_, X.shape[1])
    logger.info("Features: %s", selected_features)

    return {
        'selected_features': selected_features,
        'n_features_optimal': rfecv.n_features_,
        'ranking': rfecv.ranking_,
        'cv_results': rfecv.cv_results_,
        'selector': rfecv,
    }


def permutation_importance_selection(model, X, y, threshold=0.001, n_repeats=10,
                                     random_state=42, n_jobs=-1):
    """
    Selección de features basada en importancia por permutación.

    A diferencia de feature_importances_ (MDI), la importancia por permutación:
    - No tiene sesgo hacia features con alta cardinalidad.
    - Captura la importancia real para la predicción.
    - Funciona con cualquier modelo, no solo tree-based.

    Parámetros:
        model: Modelo ya entrenado (fitted).
        X: Features de prueba (DataFrame o ndarray).
        y: Target de prueba (Series o ndarray).
        threshold (float): Importancia mínima para conservar una feature.
        n_repeats (int): Número de repeticiones de la permutación.
        random_state (int): Semilla de reproducibilidad.
        n_jobs (int): Número de trabajos paralelos.

    Retorna:
        dict: Diccionario con:
            - 'selected_features': Nombres/índices de features seleccionadas.
            - 'importances_mean': Importancia media de cada feature.
            - 'importances_std': Desviación estándar de la importancia.
            - 'feature_ranking': DataFrame con ranking ordenado.
    """
    logger.info("Calculando importancia por permutación...")
    logger.info("  Features: %d, Repeticiones: %d, Umbral: %.4f",
                X.shape[1], n_repeats, threshold)

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Crear ranking
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False)

    # Seleccionar features sobre el umbral
    selected_mask = result.importances_mean >= threshold
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selected_mask].tolist()
    else:
        selected_features = np.where(selected_mask)[0].tolist()

    logger.info("Features seleccionadas (importancia >= %.4f): %d de %d",
                threshold, len(selected_features), X.shape[1])

    return {
        'selected_features': selected_features,
        'importances_mean': result.importances_mean,
        'importances_std': result.importances_std,
        'feature_ranking': ranking_df,
    }


def compare_feature_methods(model_factory, X, y, cv=3, random_state=42):
    """
    Compara tres métodos de selección de features y genera una tabla resumen.

    Los métodos comparados son:
    1. SelectFromModel (threshold='median') — ya usado en el proyecto.
    2. RFECV — recursive feature elimination.
    3. Permutation Importance — importancia por permutación.

    Parámetros:
        model_factory (callable): Función que retorna un nuevo estimator.
            Ejemplo: lambda: RandomForestClassifier(n_estimators=100, random_state=42)
        X: Features (DataFrame).
        y: Target (Series).
        cv (int): Folds para validación cruzada.
        random_state (int): Semilla de reproducibilidad.

    Retorna:
        pd.DataFrame: Tabla comparativa con métodos como filas y métricas como columnas.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import cross_val_score

    logger.info("Comparando métodos de selección de features...")

    results = []

    # Método 1: SelectFromModel
    model1 = model_factory()
    model1.fit(X, y)
    sfm = SelectFromModel(model1, threshold='median', prefit=True)
    X_sfm = sfm.transform(X)
    n_sfm = X_sfm.shape[1]
    score_sfm = cross_val_score(
        model_factory(), X_sfm, y, cv=cv, scoring='f1_macro'
    ).mean()
    results.append({
        'Método': 'SelectFromModel (median)',
        'Features seleccionadas': n_sfm,
        'F1-macro (CV)': round(score_sfm, 4),
    })

    # Método 2: RFECV
    rfe_result = recursive_feature_elimination(
        model_factory(), X, y, cv=cv, random_state=random_state
    )
    n_rfe = rfe_result['n_features_optimal']
    X_rfe = X.iloc[:, rfe_result['selector'].support_] if isinstance(X, pd.DataFrame) else X[:, rfe_result['selector'].support_]
    score_rfe = cross_val_score(
        model_factory(), X_rfe, y, cv=cv, scoring='f1_macro'
    ).mean()
    results.append({
        'Método': 'RFECV',
        'Features seleccionadas': n_rfe,
        'F1-macro (CV)': round(score_rfe, 4),
    })

    # Método 3: Permutation Importance
    model3 = model_factory()
    model3.fit(X, y)
    perm_result = permutation_importance_selection(
        model3, X, y, random_state=random_state
    )
    n_perm = len(perm_result['selected_features'])
    if isinstance(X, pd.DataFrame):
        X_perm = X[perm_result['selected_features']]
    else:
        X_perm = X[:, perm_result['selected_features']]
    score_perm = cross_val_score(
        model_factory(), X_perm, y, cv=cv, scoring='f1_macro'
    ).mean()
    results.append({
        'Método': 'Permutation Importance',
        'Features seleccionadas': n_perm,
        'F1-macro (CV)': round(score_perm, 4),
    })

    comparison_df = pd.DataFrame(results)
    logger.info("\n--- Comparación de Métodos ---\n%s", comparison_df.to_string(index=False))

    return comparison_df
