"""
Tests unitarios para src/feature_engineering.py
"""

import numpy as np
import pandas as pd
from src.feature_engineering import (
    drop_manual_columns,
    perform_train_test_split,
    scale_features,
    split_features_target,
)


class TestSplitFeaturesTarget:
    """Tests para la función split_features_target."""

    def test_split_features_target_separates_correctly(self):
        """Verifica que X e y se separan correctamente."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'Label': ['A', 'B', 'A'],
        })
        X, y = split_features_target(df, target_column='Label')
        assert 'Label' not in X.columns
        assert list(y) == ['A', 'B', 'A']
        assert X.shape == (3, 2)

    def test_split_features_target_default_column(self):
        """Verifica que el target por defecto es 'Label'."""
        df = pd.DataFrame({
            'a': [1, 2],
            'Label': ['X', 'Y'],
        })
        X, y = split_features_target(df)
        assert 'Label' not in X.columns
        assert len(y) == 2


class TestPerformTrainTestSplit:
    """Tests para la función perform_train_test_split."""

    def test_stratified_split_preserves_proportions(self):
        """Verifica que la estratificación preserva proporciones."""
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame({'f1': np.random.randn(n), 'f2': np.random.randn(n)})
        # 80% clase A, 20% clase B
        y = pd.Series(['A'] * 800 + ['B'] * 200)

        X_train, X_test, y_train, y_test = perform_train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Verificar que las proporciones se mantienen
        train_ratio = (y_train == 'B').mean()
        test_ratio = (y_test == 'B').mean()
        assert abs(train_ratio - 0.2) < 0.05, \
            f"Proporción en train ({train_ratio:.2f}) difiere mucho de 0.2"
        assert abs(test_ratio - 0.2) < 0.05, \
            f"Proporción en test ({test_ratio:.2f}) difiere mucho de 0.2"

    def test_split_sizes(self):
        """Verifica que los tamaños de train/test son correctos."""
        n = 100
        X = pd.DataFrame({'f1': range(n)})
        y = pd.Series(['A'] * 50 + ['B'] * 50)

        X_train, X_test, y_train, y_test = perform_train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30


class TestScaleFeatures:
    """Tests para la función scale_features."""

    def test_scale_features_zero_mean(self):
        """Verifica que el escalado produce media ~0 en train."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 10 + 50
        X_test = np.random.randn(30, 5) * 10 + 50

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # La media de X_train_scaled debe ser ~0
        means = np.mean(X_train_scaled, axis=0)
        for i, m in enumerate(means):
            assert abs(m) < 0.1, \
                f"Media de la feature {i} es {m}, debería ser ~0"

    def test_scale_features_unit_variance(self):
        """Verifica que el escalado produce varianza ~1 en train."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3) * 5 + 20
        X_test = np.random.randn(30, 3) * 5 + 20

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        stds = np.std(X_train_scaled, axis=0)
        for i, s in enumerate(stds):
            assert abs(s - 1.0) < 0.15, \
                f"Std de la feature {i} es {s}, debería ser ~1"


class TestDropManualColumns:
    """Tests para la función drop_manual_columns."""

    def test_drops_existing_columns(self):
        """Verifica que se eliminan columnas existentes."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        result = drop_manual_columns(df, ['a', 'c'])
        assert list(result.columns) == ['b']

    def test_ignores_nonexistent_columns(self):
        """Verifica que no falla si se piden columnas inexistentes."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = drop_manual_columns(df, ['c', 'd'])
        assert list(result.columns) == ['a', 'b']
