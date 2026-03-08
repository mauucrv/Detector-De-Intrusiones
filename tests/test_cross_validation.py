"""
Tests unitarios para src/cross_validation.py
"""


from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.cross_validation import cross_validate_model


class TestCrossValidateModel:
    """Tests para la función cross_validate_model."""

    def _make_data(self, n_samples=500, n_classes=3):
        """Genera datos sintéticos para testing."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_classes=n_classes,
            n_informative=6,
            random_state=42,
        )
        return X, y

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene todas las claves esperadas."""
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = cross_validate_model(model, X, y, n_splits=3)

        expected_keys = [
            'accuracy_mean', 'accuracy_std',
            'precision_mean', 'precision_std',
            'recall_mean', 'recall_std',
            'f1_mean', 'f1_std',
            'scores_per_fold',
        ]
        for key in expected_keys:
            assert key in results, f"Falta la clave '{key}' en los resultados"

    def test_scores_per_fold_has_correct_length(self):
        """Verifica que scores_per_fold tiene el número correcto de folds."""
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        n_splits = 4
        results = cross_validate_model(model, X, y, n_splits=n_splits)

        for metric, scores in results['scores_per_fold'].items():
            assert len(scores) == n_splits, \
                f"'{metric}' tiene {len(scores)} folds, esperados {n_splits}"

    def test_accuracy_is_reasonable(self):
        """Verifica que la accuracy es razonable (> 0.5 para un RF)."""
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = cross_validate_model(model, X, y, n_splits=3)

        assert results['accuracy_mean'] > 0.5, \
            f"Accuracy demasiado baja: {results['accuracy_mean']:.4f}"

    def test_metrics_within_valid_range(self):
        """Verifica que todas las métricas están en [0, 1]."""
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = cross_validate_model(model, X, y, n_splits=3)

        for key in ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']:
            value = results[key]
            assert 0 <= value <= 1, \
                f"'{key}' fuera de rango [0, 1]: {value}"

    def test_std_is_non_negative(self):
        """Verifica que las desviaciones estándar no son negativas."""
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = cross_validate_model(model, X, y, n_splits=3)

        for key in ['accuracy_std', 'precision_std', 'recall_std', 'f1_std']:
            value = results[key]
            assert value >= 0, f"'{key}' es negativo: {value}"

    def test_reproducibility_with_same_seed(self):
        """Verifica que los resultados son reproducibles con la misma semilla."""
        X, y = self._make_data()
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)

        results1 = cross_validate_model(model1, X, y, n_splits=3, random_state=42)
        results2 = cross_validate_model(model2, X, y, n_splits=3, random_state=42)

        assert abs(results1['accuracy_mean'] - results2['accuracy_mean']) < 1e-10, \
            "Los resultados no son reproducibles"
