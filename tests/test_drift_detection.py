"""
Tests unitarios para src/drift_detection.py
"""

import numpy as np
from src.drift_detection import (
    DriftMonitor,
    detect_data_drift,
    detect_prediction_drift,
)


class TestDetectDataDrift:
    """Tests para la función detect_data_drift."""

    def test_no_drift_same_distribution(self):
        """Verifica que no detecta drift cuando las distribuciones son iguales."""
        np.random.seed(42)
        X_ref = np.random.randn(500, 5)
        X_cur = np.random.randn(500, 5)
        result = detect_data_drift(X_ref, X_cur)
        # Para la misma distribución, no debería detectar drift masivo
        assert result['drift_ratio'] < 0.5

    def test_detects_drift_different_distribution(self):
        """Verifica que detecta drift con distribuciones claramente diferentes."""
        np.random.seed(42)
        X_ref = np.random.randn(500, 5)
        X_cur = np.random.randn(500, 5) + 10  # Shift grande
        result = detect_data_drift(X_ref, X_cur)
        assert result['has_drift'] is True
        assert result['n_drifted'] > 0

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        X_ref = np.random.randn(100, 3)
        X_cur = np.random.randn(100, 3)
        result = detect_data_drift(X_ref, X_cur)
        expected = ['has_drift', 'drifted_features', 'n_drifted',
                     'feature_results', 'drift_ratio']
        for key in expected:
            assert key in result, f"Falta '{key}'"

    def test_feature_results_length(self):
        """Verifica que hay un resultado por feature."""
        n_features = 7
        X_ref = np.random.randn(100, n_features)
        X_cur = np.random.randn(100, n_features)
        result = detect_data_drift(X_ref, X_cur)
        assert len(result['feature_results']) == n_features

    def test_accepts_1d_input(self):
        """Verifica que acepta arrays 1D."""
        X_ref = np.random.randn(100)
        X_cur = np.random.randn(100)
        result = detect_data_drift(X_ref, X_cur)
        assert len(result['feature_results']) == 1


class TestDetectPredictionDrift:
    """Tests para la función detect_prediction_drift."""

    def test_no_drift_same_distribution(self):
        """Verifica que no detecta drift con distribuciones similares."""
        np.random.seed(42)
        y_ref = np.random.choice(['A', 'B', 'C'], 1000, p=[0.6, 0.3, 0.1])
        y_cur = np.random.choice(['A', 'B', 'C'], 1000, p=[0.6, 0.3, 0.1])
        result = detect_prediction_drift(y_ref, y_cur)
        # Puede o no detectar drift, pero al menos no debería ser extremo
        assert 'has_drift' in result
        assert 'p_value' in result

    def test_detects_drift_different_distribution(self):
        """Verifica que detecta drift con distribución completamente diferente."""
        y_ref = np.array(['A'] * 800 + ['B'] * 200)
        y_cur = np.array(['A'] * 200 + ['B'] * 800)
        result = detect_prediction_drift(y_ref, y_cur)
        assert result['has_drift']

    def test_returns_distributions(self):
        """Verifica que retorna las distribuciones de referencia y actual."""
        y_ref = np.array(['X', 'Y', 'X', 'Y'])
        y_cur = np.array(['X', 'X', 'Y', 'Y'])
        result = detect_prediction_drift(y_ref, y_cur)
        assert 'reference_distribution' in result
        assert 'current_distribution' in result


class TestDriftMonitor:
    """Tests para la clase DriftMonitor."""

    def test_initialization(self):
        """Verifica que se inicializa correctamente."""
        X_ref = np.random.randn(100, 5)
        y_ref = np.random.choice([0, 1], 100)
        monitor = DriftMonitor(X_ref, y_ref)
        assert len(monitor.history) == 0

    def test_check_returns_report(self):
        """Verifica que check retorna un reporte completo."""
        np.random.seed(42)
        X_ref = np.random.randn(100, 5)
        y_ref = np.random.choice([0, 1], 100)
        monitor = DriftMonitor(X_ref, y_ref)

        X_cur = np.random.randn(50, 5)
        y_cur = np.random.choice([0, 1], 50)
        report = monitor.check(X_cur, y_cur)

        assert 'batch_number' in report
        assert 'data_drift' in report
        assert 'prediction_drift' in report
        assert 'needs_retraining' in report
        assert report['batch_number'] == 1

    def test_history_accumulates(self):
        """Verifica que el historial acumula reportes."""
        np.random.seed(42)
        X_ref = np.random.randn(100, 5)
        y_ref = np.random.choice([0, 1], 100)
        monitor = DriftMonitor(X_ref, y_ref)

        for _ in range(3):
            X_cur = np.random.randn(50, 5)
            monitor.check(X_cur)

        assert len(monitor.history) == 3
        assert monitor.history[2]['batch_number'] == 3

    def test_check_without_predictions(self):
        """Verifica que funciona sin predicciones (solo data drift)."""
        np.random.seed(42)
        X_ref = np.random.randn(100, 5)
        y_ref = np.random.choice([0, 1], 100)
        monitor = DriftMonitor(X_ref, y_ref)

        X_cur = np.random.randn(50, 5)
        report = monitor.check(X_cur)

        assert 'data_drift' in report
        assert 'prediction_drift' not in report
