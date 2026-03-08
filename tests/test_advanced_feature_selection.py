"""
Tests unitarios para src/advanced_feature_selection.py
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.advanced_feature_selection import (
    permutation_importance_selection,
    recursive_feature_elimination,
)


def _make_data(n_features=15, n_informative=5):
    """Crea datos sintéticos con features informativas y ruido."""
    X, y = make_classification(
        n_samples=200, n_features=n_features,
        n_informative=n_informative, n_redundant=3,
        random_state=42,
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=columns)
    return X_df, y


class TestRecursiveFeatureElimination:
    """Tests para recursive_feature_elimination."""

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = recursive_feature_elimination(model, X, y, cv=2, min_features=3)
        expected = ['selected_features', 'n_features_optimal', 'ranking',
                     'cv_results', 'selector']
        for key in expected:
            assert key in result, f"Falta '{key}'"

    def test_selects_fewer_features(self):
        """Verifica que se seleccionan menos features que las originales."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = recursive_feature_elimination(model, X, y, cv=2, min_features=3)
        assert result['n_features_optimal'] <= X.shape[1]
        assert result['n_features_optimal'] >= 3

    def test_selected_features_are_valid_names(self):
        """Verifica que los nombres de features seleccionadas son válidos."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = recursive_feature_elimination(model, X, y, cv=2, min_features=3)
        for feat in result['selected_features']:
            assert feat in X.columns


class TestPermutationImportanceSelection:
    """Tests para permutation_importance_selection."""

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = permutation_importance_selection(model, X, y, n_repeats=3)
        expected = ['selected_features', 'importances_mean',
                     'importances_std', 'feature_ranking']
        for key in expected:
            assert key in result, f"Falta '{key}'"

    def test_feature_ranking_is_sorted(self):
        """Verifica que el ranking está ordenado por importancia descendente."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = permutation_importance_selection(model, X, y, n_repeats=3)
        ranking = result['feature_ranking']
        assert ranking['importance_mean'].is_monotonic_decreasing

    def test_threshold_filters_features(self):
        """Verifica que el umbral filtra features correctamente."""
        X, y = _make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        # Con umbral alto, se seleccionan menos features
        result_high = permutation_importance_selection(
            model, X, y, threshold=0.1, n_repeats=3
        )
        result_low = permutation_importance_selection(
            model, X, y, threshold=0.0, n_repeats=3
        )
        assert len(result_high['selected_features']) <= len(result_low['selected_features'])
