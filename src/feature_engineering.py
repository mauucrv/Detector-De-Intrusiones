"""
Funciones de feature engineering y preprocesamiento.

Funciones extraídas del notebook 02_feature_engineering_and_preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def drop_manual_columns(df, columns_to_drop):
    """
    Elimina manualmente las columnas especificadas del DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        columns_to_drop (list): Lista de nombres de columnas a eliminar.

    Retorna:
        pd.DataFrame: DataFrame sin las columnas especificadas.
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_cols)
    print(f"Columnas eliminadas manualmente: {existing_cols}")
    print(f"Dimensiones resultantes: {df_cleaned.shape}")
    return df_cleaned


def split_features_target(df, target_column='Label'):
    """
    Separa el DataFrame en features (X) y target (y).

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        target_column (str): Nombre de la columna objetivo.

    Retorna:
        tuple: (X, y) donde X son las features y y es el target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def perform_train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Realiza la partición estratificada de train/test.

    Parámetros:
        X: Features.
        y: Target.
        test_size (float): Proporción para test.
        random_state (int): Semilla de reproducibilidad.

    Retorna:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Dimensiones de entrenamiento: X={X_train.shape}, y={y_train.shape}")
    print(f"Dimensiones de prueba: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test


def select_features_with_rf(X_train, y_train, X_test, random_state=42, threshold='median'):
    """
    Selecciona las features más importantes usando un RandomForestClassifier
    con SelectFromModel.

    Parámetros:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de prueba.
        random_state (int): Semilla de reproducibilidad.
        threshold (str): Umbral para SelectFromModel.

    Retorna:
        tuple: (X_train_selected, X_test_selected, selected_feature_names)
    """
    print("Entrenando Random Forest para selección de features...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_selector.fit(X_train, y_train)

    selector = SelectFromModel(rf_selector, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    selected_feature_names = X_train.columns[selector.get_support()].tolist()

    print(f"Features originales: {X_train.shape[1]}")
    print(f"Features seleccionadas: {len(selected_feature_names)}")
    print(f"Features: {selected_feature_names}")

    return X_train_selected, X_test_selected, selected_feature_names


def scale_features(X_train, X_test):
    """
    Escala las features usando StandardScaler (fit en train, transform en ambos).

    Parámetros:
        X_train: Features de entrenamiento.
        X_test: Features de prueba.

    Retorna:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Escalado completado.")
    return X_train_scaled, X_test_scaled, scaler
