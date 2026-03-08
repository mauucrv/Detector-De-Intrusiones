"""
Tests unitarios para src/data_utils.py
"""

import numpy as np
import pandas as pd
from src.data_utils import clean_dataframe, optimize_memory


class TestCleanDataframe:
    """Tests para la función clean_dataframe."""

    def _make_df(self):
        """Crea un DataFrame de prueba con problemas típicos."""
        return pd.DataFrame({
            ' Column A ': [1.0, 2.0, np.inf, 4.0, 5.0],
            'Column B': [10.0, -np.inf, 30.0, 40.0, 50.0],
            'Label': ['BENIGN', 'Attack�Type', 'BENIGN', 'BENIGN', 'DDoS'],
        })

    def test_clean_dataframe_strips_column_names(self):
        """Verifica que se eliminan espacios en nombres de columnas."""
        df = self._make_df()
        cleaned = clean_dataframe(df)
        for col in cleaned.columns:
            assert col == col.strip(), f"La columna '{col}' tiene espacios"

    def test_clean_dataframe_removes_inf(self):
        """Verifica que se eliminan filas con valores infinitos."""
        df = self._make_df()
        cleaned = clean_dataframe(df)
        assert not np.any(np.isinf(cleaned.select_dtypes(include=[np.number]).values)), \
            "Se encontraron valores infinitos después de la limpieza"

    def test_clean_dataframe_removes_nan(self):
        """Verifica que se eliminan filas con NaN."""
        df = self._make_df()
        cleaned = clean_dataframe(df)
        assert not cleaned.isnull().any().any(), \
            "Se encontraron valores NaN después de la limpieza"

    def test_clean_dataframe_fixes_label_encoding(self):
        """Verifica que se corrigen caracteres mal codificados en Label."""
        df = self._make_df()
        cleaned = clean_dataframe(df)
        for label in cleaned['Label']:
            assert '�' not in label, \
                f"Se encontró carácter mal codificado en: {label}"

    def test_clean_dataframe_reduces_rows(self):
        """Verifica que se eliminaron las filas con inf/NaN."""
        df = self._make_df()
        original_len = len(df)
        cleaned = clean_dataframe(df)
        assert len(cleaned) < original_len, \
            "El DataFrame limpio debería tener menos filas"


class TestOptimizeMemory:
    """Tests para la función optimize_memory."""

    def test_optimize_memory_reduces_usage(self):
        """Verifica que la optimización reduce el uso de memoria."""
        # Crear un DataFrame con tipos amplios
        np.random.seed(42)
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, size=1000).astype(np.int64),
            'float_col': np.random.random(1000).astype(np.float64),
            'str_col': np.random.choice(['A', 'B', 'C'], size=1000),
        })
        mem_before = df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(df)
        mem_after = df_opt.memory_usage(deep=True).sum()
        assert mem_after < mem_before, \
            f"La memoria no se redujo: {mem_before} -> {mem_after}"

    def test_optimize_memory_preserves_values(self):
        """Verifica que los valores se preservan tras la optimización."""
        np.random.seed(42)
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, size=100).astype(np.int64),
        })
        original_values = df['int_col'].values.copy()
        df_opt = optimize_memory(df)
        np.testing.assert_array_equal(
            df_opt['int_col'].values, original_values,
            err_msg="Los valores cambiaron después de la optimización"
        )
