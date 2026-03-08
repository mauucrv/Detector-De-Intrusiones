"""
Tests unitarios para src/inference.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.inference import IntrusionDetector
from src.model_persistence import save_model


def _create_test_model(tmpdir):
    """Crea un modelo de prueba y lo guarda, retornando la ruta."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    path = os.path.join(tmpdir, 'test_model.joblib')
    save_model(model, path, metadata={'name': 'test'})
    return path, X, y, model


class TestIntrusionDetector:
    """Tests para la clase IntrusionDetector."""

    def test_predict_returns_array(self):
        """Verifica que predict devuelve un ndarray."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, X, _, _ = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)
            predictions = detector.predict(X)
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(X)

    def test_predict_with_dataframe(self):
        """Verifica que predict acepta DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, X, _, _ = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)
            df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            predictions = detector.predict(df)
            assert len(predictions) == len(df)

    def test_predict_proba_returns_probabilities(self):
        """Verifica que predict_proba devuelve probabilidades válidas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, X, _, _ = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)
            proba = detector.predict_proba(X)

            # Cada fila debe sumar ~1
            row_sums = np.sum(proba, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

            # Todos los valores deben estar en [0, 1]
            assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_matches_original_model(self):
        """Verifica que las predicciones coinciden con el modelo original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, X, _, model = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)

            original_pred = model.predict(X)
            detector_pred = detector.predict(X)

            np.testing.assert_array_equal(original_pred, detector_pred)

    def test_get_classes(self):
        """Verifica que get_classes retorna las clases del modelo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, _, _, model = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)
            classes = detector.get_classes()
            np.testing.assert_array_equal(classes, model.classes_)

    def test_nonexistent_model_raises_error(self):
        """Verifica que un modelo inexistente lanza FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            IntrusionDetector('/nonexistent/model.joblib')

    def test_predict_with_csv_file(self):
        """Verifica que predict acepta una ruta a CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, X, _, _ = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)

            csv_path = os.path.join(tmpdir, 'test_data.csv')
            df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            df.to_csv(csv_path, index=False)

            predictions = detector.predict(csv_path)
            assert len(predictions) == len(X)

    def test_predict_invalid_type_raises_error(self):
        """Verifica que un tipo de dato inválido lanza TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path, _, _, _ = _create_test_model(tmpdir)
            detector = IntrusionDetector(path)
            with pytest.raises(TypeError):
                detector.predict(12345)
