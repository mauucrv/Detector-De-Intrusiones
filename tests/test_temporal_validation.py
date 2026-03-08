"""
Tests unitarios para src/temporal_validation.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.exceptions import InvalidDataError
from src.temporal_validation import (
    add_day_column,
    temporal_train_test_split,
    walk_forward_validate,
)


def _make_temporal_df(n_per_day=50, n_days=5):
    """Crea un DataFrame sintético con columna Day y features."""
    np.random.seed(42)
    dfs = []
    for day in range(n_days):
        df_day = pd.DataFrame({
            'f1': np.random.randn(n_per_day) + day * 0.5,
            'f2': np.random.randn(n_per_day),
            'Day': day,
            'Label': np.random.choice(['BENIGN', 'Attack'], n_per_day, p=[0.8, 0.2]),
        })
        dfs.append(df_day)
    return pd.concat(dfs, ignore_index=True)


class TestAddDayColumn:
    """Tests para la función add_day_column."""

    def test_maps_day_names_to_indices(self):
        """Verifica que mapea nombres de día a índices."""
        df = pd.DataFrame({
            'DayName': ['Monday', 'Tuesday', 'Wednesday'],
            'f1': [1, 2, 3],
        })
        result = add_day_column(df, source_column='DayName')
        assert list(result['Day']) == [0, 1, 2]

    def test_preserves_existing_day_column(self):
        """Verifica que no modifica un Day existente."""
        df = pd.DataFrame({'Day': [0, 1, 2], 'f1': [1, 2, 3]})
        result = add_day_column(df)
        assert list(result['Day']) == [0, 1, 2]

    def test_raises_on_missing_source_column(self):
        """Verifica que lanza error si no hay columna de día."""
        df = pd.DataFrame({'f1': [1, 2, 3]})
        with pytest.raises(InvalidDataError):
            add_day_column(df)


class TestTemporalTrainTestSplit:
    """Tests para la función temporal_train_test_split."""

    def test_split_sizes(self):
        """Verifica que los splits tienen las muestras correctas."""
        df = _make_temporal_df()
        df_train, df_test = temporal_train_test_split(df, train_days=[0, 1, 2], test_days=[3, 4])
        assert len(df_train) == 150  # 3 days * 50
        assert len(df_test) == 100   # 2 days * 50

    def test_no_overlap(self):
        """Verifica que no hay solapamiento entre train y test."""
        df = _make_temporal_df()
        df_train, df_test = temporal_train_test_split(df)
        train_indices = set(df_train.index)
        test_indices = set(df_test.index)
        assert len(train_indices & test_indices) == 0

    def test_raises_on_missing_column(self):
        """Verifica que lanza error si no existe la columna Day."""
        df = pd.DataFrame({'f1': [1, 2, 3]})
        with pytest.raises(InvalidDataError):
            temporal_train_test_split(df)

    def test_raises_on_empty_split(self):
        """Verifica que lanza error si un split queda vacío."""
        df = _make_temporal_df(n_days=3)
        with pytest.raises(InvalidDataError):
            temporal_train_test_split(df, train_days=[0, 1, 2], test_days=[5, 6])


class TestWalkForwardValidate:
    """Tests para la función walk_forward_validate."""

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        df = _make_temporal_df()
        result = walk_forward_validate(
            model_factory=lambda: RandomForestClassifier(n_estimators=5, random_state=42),
            df=df, min_train_days=2,
        )
        assert 'steps' in result
        assert 'mean_score' in result
        assert 'std_score' in result

    def test_number_of_steps(self):
        """Verifica el número correcto de pasos de walk-forward."""
        df = _make_temporal_df(n_days=5)
        result = walk_forward_validate(
            model_factory=lambda: RandomForestClassifier(n_estimators=5, random_state=42),
            df=df, min_train_days=2,
        )
        # 5 days, min 2 train → steps for days 2, 3, 4 = 3 steps
        assert len(result['steps']) == 3

    def test_scores_in_valid_range(self):
        """Verifica que los scores están en [0, 1]."""
        df = _make_temporal_df()
        result = walk_forward_validate(
            model_factory=lambda: RandomForestClassifier(n_estimators=5, random_state=42),
            df=df, min_train_days=2,
        )
        assert 0 <= result['mean_score'] <= 1
        for step in result['steps']:
            assert 0 <= step['score'] <= 1
