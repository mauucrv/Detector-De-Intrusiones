"""
Utilidades para la carga, limpieza y optimización de datos.

Funciones extraídas del notebook 00_data_ingestion_and_optimization.
"""

import pandas as pd
import numpy as np
import os
import glob


def load_csv_files(path_to_csvs):
    """
    Carga todos los archivos CSV de un directorio y los concatena en un solo DataFrame.

    Parámetros:
        path_to_csvs (str): Ruta al directorio que contiene los archivos CSV.

    Retorna:
        pd.DataFrame: DataFrame combinado con todos los datos.
    """
    csv_files = glob.glob(os.path.join(path_to_csvs, "*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Dataset combinado cargado. Dimensiones: {df.shape}")
    return df


def clean_dataframe(df):
    """
    Aplica la secuencia de limpieza de datos:
    1. Limpia nombres de columnas (elimina espacios en blanco).
    2. Reemplaza valores infinitos por NaN y elimina las filas afectadas.
    3. Limpia la columna 'Label' (corrige caracteres mal codificados).

    Parámetros:
        df (pd.DataFrame): DataFrame a limpiar.

    Retorna:
        pd.DataFrame: DataFrame limpio.
    """
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if 'Label' in df.columns:
        df['Label'] = (df['Label']
                       .str.replace('�', ' ', regex=False)
                       .str.replace(r'\s+', ' ', regex=True)
                       .str.strip())

    return df


def optimize_memory(df):
    """
    Itera sobre todas las columnas de un DataFrame y modifica los tipos de datos
    para reducir el uso de memoria, aplicando 'downcasting' a tipos numéricos
    y convirtiendo tipos 'object' a 'category'.

    Parámetros:
        df (pd.DataFrame): DataFrame a optimizar.

    Retorna:
        pd.DataFrame: DataFrame con tipos de datos optimizados.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Uso de memoria inicial: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
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
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Uso de memoria final: {end_mem:.2f} MB')
    print(f'Reducción del {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df
