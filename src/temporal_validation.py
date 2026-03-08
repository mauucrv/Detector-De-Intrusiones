"""
Módulo de validación temporal para el Detector de Intrusiones.

El dataset CIC-IDS-2017 contiene tráfico de 5 días (lunes a viernes).
En un IDS real, siempre se predice tráfico futuro, por lo que un split
temporal es más realista que un split aleatorio estratificado.

Este módulo implementa:
- Split temporal por día (entrenar en días anteriores, validar en posteriores).
- Walk-forward validation (validación progresiva).
"""

import logging

import numpy as np
from sklearn.metrics import f1_score

from src.exceptions import InvalidDataError

logger = logging.getLogger(__name__)

# Mapeo de archivos CIC-IDS-2017 a días de la semana
CIC_IDS_DAY_MAP = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
}


def add_day_column(df, source_column=None, day_map=None):
    """
    Agrega una columna 'Day' al DataFrame basada en la fuente de los datos.

    Si el DataFrame tiene una columna con información del día (por ejemplo,
    derivada del nombre del archivo CSV original), la usa para asignar un
    índice numérico de día.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        source_column (str, opcional): Nombre de la columna que contiene
            la información del día. Si es None, intenta detectarla.
        day_map (dict, opcional): Mapeo de nombres de día a índices numéricos.
            Por defecto usa CIC_IDS_DAY_MAP.

    Retorna:
        pd.DataFrame: DataFrame con columna 'Day' agregada.

    Raises:
        InvalidDataError: Si no se puede determinar el día.
    """
    df = df.copy()
    day_map = day_map or CIC_IDS_DAY_MAP

    if source_column and source_column in df.columns:
        df['Day'] = df[source_column].map(day_map)
        if df['Day'].isna().any():
            unknown = df[df['Day'].isna()][source_column].unique()
            raise InvalidDataError(
                f"Valores no reconocidos en columna '{source_column}': {unknown}. "
                f"Valores esperados: {list(day_map.keys())}"
            )
    elif 'Day' in df.columns:
        logger.info("Columna 'Day' ya existe en el DataFrame.")
    else:
        raise InvalidDataError(
            "No se encontró una columna de día. "
            "Proporcione 'source_column' o agregue una columna 'Day' manualmente."
        )

    return df


def temporal_train_test_split(df, day_column='Day', train_days=None, test_days=None):
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose
    en los días, simulando un escenario realista de producción.

    Parámetros:
        df (pd.DataFrame): DataFrame con columna de día.
        day_column (str): Nombre de la columna que contiene el índice de día.
        train_days (list[int], opcional): Días para entrenamiento.
            Por defecto: [0, 1, 2] (lunes a miércoles).
        test_days (list[int], opcional): Días para prueba.
            Por defecto: [3, 4] (jueves y viernes).

    Retorna:
        tuple: (df_train, df_test)

    Raises:
        InvalidDataError: Si la columna de día no existe o los días no son válidos.
    """
    if day_column not in df.columns:
        raise InvalidDataError(
            f"La columna '{day_column}' no existe en el DataFrame. "
            "Use add_day_column() primero."
        )

    train_days = train_days if train_days is not None else [0, 1, 2]
    test_days = test_days if test_days is not None else [3, 4]

    available_days = sorted(df[day_column].unique())
    logger.info("Días disponibles en el dataset: %s", available_days)
    logger.info("Días de entrenamiento: %s", train_days)
    logger.info("Días de prueba: %s", test_days)

    df_train = df[df[day_column].isin(train_days)].copy()
    df_test = df[df[day_column].isin(test_days)].copy()

    if df_train.empty:
        raise InvalidDataError(f"No hay datos para los días de entrenamiento: {train_days}")
    if df_test.empty:
        raise InvalidDataError(f"No hay datos para los días de prueba: {test_days}")

    logger.info("Split temporal: train=%d registros, test=%d registros",
                len(df_train), len(df_test))

    return df_train, df_test


def walk_forward_validate(model_factory, df, day_column='Day',
                          target_column='Label', feature_columns=None,
                          min_train_days=2, scoring='f1_macro'):
    """
    Realiza validación walk-forward (progresiva) sobre los datos temporales.

    En cada paso, entrena con todos los días anteriores y valida en el día actual.
    Esto simula cómo funcionaría el modelo en producción, donde se entrena
    con datos históricos y se predice tráfico nuevo.

    Parámetros:
        model_factory (callable): Función que retorna una nueva instancia del modelo.
            Ejemplo: lambda: RandomForestClassifier(n_estimators=100)
        df (pd.DataFrame): DataFrame con columna de día y features.
        day_column (str): Nombre de la columna de día.
        target_column (str): Nombre de la columna objetivo.
        feature_columns (list, opcional): Columnas de features. Si es None,
            usa todas las columnas excepto day_column y target_column.
        min_train_days (int): Mínimo de días necesarios para entrenar.
        scoring (str): Métrica para reportar.

    Retorna:
        dict: Diccionario con resultados por paso:
            - 'steps': Lista de dicts con info de cada paso (train_days, test_day, score).
            - 'mean_score': Score promedio a lo largo de los pasos.
            - 'std_score': Desviación estándar del score.
    """
    if day_column not in df.columns:
        raise InvalidDataError(f"La columna '{day_column}' no existe en el DataFrame.")
    if target_column not in df.columns:
        raise InvalidDataError(f"La columna '{target_column}' no existe en el DataFrame.")

    days = sorted(df[day_column].unique())

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in [day_column, target_column]]

    logger.info("Iniciando walk-forward validation...")
    logger.info("Días totales: %s, min_train_days: %d", days, min_train_days)

    steps = []
    scores = []

    for i in range(min_train_days, len(days)):
        train_days = days[:i]
        test_day = days[i]

        train_mask = df[day_column].isin(train_days)
        test_mask = df[day_column] == test_day

        X_train = df.loc[train_mask, feature_columns]
        y_train = df.loc[train_mask, target_column]
        X_test = df.loc[test_mask, feature_columns]
        y_test = df.loc[test_mask, target_column]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if scoring == 'f1_macro':
            score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        else:
            score = model.score(X_test, y_test)

        step_result = {
            'train_days': list(train_days),
            'test_day': test_day,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'score': score,
        }
        steps.append(step_result)
        scores.append(score)

        logger.info("  Paso %d: train_days=%s, test_day=%s, score=%.4f",
                     len(steps), train_days, test_day, score)

    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    logger.info("--- Resultados Walk-Forward ---")
    logger.info("Score promedio: %.4f (+/- %.4f)", mean_score, std_score)

    return {
        'steps': steps,
        'mean_score': mean_score,
        'std_score': std_score,
    }
