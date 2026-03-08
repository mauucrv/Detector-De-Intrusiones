"""
Tests unitarios para src/model_persistence.py
"""

import os
import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.model_persistence import load_model, save_model


class TestSaveModel:
    """Tests para la función save_model."""

    def test_save_model_creates_file(self):
        """Verifica que save_model crea el archivo."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model.joblib')
            result_path = save_model(model, path)
            assert os.path.exists(path), "El archivo no se creó"
            assert os.path.isabs(result_path), "No retornó una ruta absoluta"

    def test_save_model_with_metadata(self):
        """Verifica que los metadatos se guardan correctamente."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, random_state=42)
        model.fit(X, y)

        metadata = {'name': 'test_model', 'accuracy': 0.95}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, path, metadata=metadata)
            loaded_model, loaded_metadata = load_model(path)
            assert loaded_metadata['name'] == 'test_model'
            assert loaded_metadata['accuracy'] == 0.95

    def test_save_model_creates_directories(self):
        """Verifica que se crean directorios intermedios."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'subdir', 'deep', 'model.joblib')
            save_model(model, path)
            assert os.path.exists(path)


class TestLoadModel:
    """Tests para la función load_model."""

    def test_save_and_load_roundtrip(self):
        """Verifica que un modelo se puede guardar y cargar correctamente."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=100, random_state=42)
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, path)
            loaded_model, metadata = load_model(path)
            loaded_predictions = loaded_model.predict(X)

            np.testing.assert_array_equal(
                original_predictions, loaded_predictions,
                err_msg="Las predicciones difieren tras carga"
            )

    def test_load_nonexistent_file_raises_error(self):
        """Verifica que cargar un archivo inexistente lanza FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model('/nonexistent/path/model.joblib')

    def test_load_model_includes_saved_at(self):
        """Verifica que saved_at se incluye en los metadatos."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X, y = make_classification(n_samples=50, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, path)
            _, metadata = load_model(path)
            assert 'saved_at' in metadata, "Falta 'saved_at' en metadatos"
