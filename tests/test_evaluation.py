"""
Tests unitarios para src/evaluation.py
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Backend sin GUI para tests

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import (
    compute_metrics,
    compute_train_test_gap,
    plot_confusion_matrix,
)


class TestComputeMetrics:
    """Tests para la función compute_metrics."""

    def _make_model_and_data(self):
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3,
            n_informative=6, random_state=42,
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        model, X, y = self._make_model_and_data()
        result = compute_metrics(model, X, y)
        expected_keys = ['y_pred', 'classification_report',
                         'classification_report_dict', 'confusion_matrix']
        for key in expected_keys:
            assert key in result, f"Falta la clave '{key}'"

    def test_y_pred_shape(self):
        """Verifica que y_pred tiene la longitud correcta."""
        model, X, y = self._make_model_and_data()
        result = compute_metrics(model, X, y)
        assert len(result['y_pred']) == len(y)

    def test_classification_report_dict_has_accuracy(self):
        """Verifica que el reporte dict contiene accuracy."""
        model, X, y = self._make_model_and_data()
        result = compute_metrics(model, X, y)
        assert 'accuracy' in result['classification_report_dict']

    def test_confusion_matrix_shape(self):
        """Verifica que la matriz de confusión tiene la forma correcta."""
        model, X, y = self._make_model_and_data()
        n_classes = len(np.unique(y))
        result = compute_metrics(model, X, y)
        assert result['confusion_matrix'].shape == (n_classes, n_classes)


class TestComputeTrainTestGap:
    """Tests para la función compute_train_test_gap."""

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        X, y = make_classification(n_samples=200, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_train_test_gap(model, X, y, X, y)
        for key in ['train_score', 'test_score', 'gap', 'is_overfitting']:
            assert key in result

    def test_gap_is_zero_for_same_data(self):
        """Verifica que el gap es 0 cuando train y test son los mismos datos."""
        X, y = make_classification(n_samples=200, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_train_test_gap(model, X, y, X, y)
        assert result['gap'] == 0.0

    def test_scores_in_valid_range(self):
        """Verifica que los scores están en [0, 1]."""
        X, y = make_classification(n_samples=200, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = compute_train_test_gap(model, X, y, X, y)
        assert 0 <= result['train_score'] <= 1
        assert 0 <= result['test_score'] <= 1


class TestPlotConfusionMatrix:
    """Tests para la función plot_confusion_matrix."""

    def test_returns_figure(self):
        """Verifica que se retorna una figura matplotlib."""
        import matplotlib.pyplot as plt
        cm = np.array([[50, 5], [3, 42]])
        fig = plot_confusion_matrix(cm, ['A', 'B'], 'test_model')
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_accepts_multiclass(self):
        """Verifica que funciona con múltiples clases."""
        import matplotlib.pyplot as plt
        cm = np.array([[30, 2, 1], [3, 25, 2], [1, 3, 33]])
        fig = plot_confusion_matrix(cm, ['A', 'B', 'C'], 'test_model')
        assert fig is not None
        plt.close(fig)
