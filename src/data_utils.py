"""
Utilidades para la carga, limpieza y optimización de datos.

Funciones extraídas del notebook 00_data_ingestion_and_optimization.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

from src.exceptions import InvalidDataError

logger = logging.getLogger(__name__)


def load_csv_files(path_to_csvs):
    """
    Carga todos los archivos CSV de un directorio y los concatena en un solo DataFrame.

    Parámetros:
        path_to_csvs (str): Ruta al directorio que contiene los archivos CSV.

    Retorna:
        pd.DataFrame: DataFrame combinado con todos los datos.

    Raises:
        InvalidDataError: Si no se encontraron archivos CSV en el directorio.
    """
    csv_files = glob.glob(os.path.join(path_to_csvs, "*.csv"))
    if not csv_files:
        raise InvalidDataError(f"No se encontraron archivos CSV en: {path_to_csvs}")

    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info("Dataset combinado cargado. Dimensiones: %s", df.shape)
    return df


def clean_dataframe(df):
    """
    Aplica la secuencia de limpieza de datos:
    1. Limpia nombres de columnas (elimina espacios en blanco).
    2. Reemplaza valores infinitos por NaN y elimina las filas afectadas.
    3. Limpia la columna 'Label' (corrige caracteres mal codificados).

    Nota: Retorna una copia; el DataFrame original no se modifica.

    Parámetros:
        df (pd.DataFrame): DataFrame a limpiar.

    Retorna:
        pd.DataFrame: DataFrame limpio.

    Raises:
        InvalidDataError: Si el DataFrame está vacío.
    """
    if df.empty:
        raise InvalidDataError("El DataFrame de entrada está vacío.")

    df = df.copy()
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info("Filas eliminadas (inf/NaN): %d", rows_dropped)

    if 'Label' in df.columns:
        df['Label'] = (df['Label']
                       .str.replace('�', ' ', regex=False)
                       .str.replace(r'\\s+', ' ', regex=True)
                       .str.strip())

    return df


def optimize_memory(df):
    """
    Itera sobre todas las columnas de un DataFrame y modifica los tipos de datos
    para reducir el uso de memoria, aplicando 'downcasting' a tipos numéricos
    y convirtiendo tipos 'object' a 'category'.

    Nota: Retorna una copia; el DataFrame original no se modifica.
    Usa float32 como tipo flotante mínimo para mantener compatibilidad con scikit-learn.

    Parámetros:
        df (pd.DataFrame): DataFrame a optimizar.

    Retorna:
        pd.DataFrame: DataFrame con tipos de datos optimizados.
    """
    df = df.copy()
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info("Uso de memoria inicial: %.2f MB", start_mem)

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type) and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    logger.info("Uso de memoria final: %.2f MB (reducción del %.1f%%)", end_mem, reduction)
    return df
