"""
Tests unitarios para src/hyperparameter_tuning.py
"""


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from src.hyperparameter_tuning import (
    get_gb_param_distributions,
    get_param_distributions,
    get_rf_param_distributions,
    tune_model,
    tune_random_forest,
)


class TestParamDistributions:
    """Tests para las funciones de distribución de parámetros."""

    def test_rf_distributions_has_required_keys(self):
        """Verifica que RF tiene los parámetros esperados."""
        params = get_rf_param_distributions()
        expected = ['n_estimators', 'max_depth', 'min_samples_split',
                     'min_samples_leaf', 'max_features']
        for key in expected:
            assert key in params, f"Falta '{key}' en RF params"

    def test_gb_distributions_has_required_keys(self):
        """Verifica que GB tiene los parámetros esperados."""
        params = get_gb_param_distributions()
        expected = ['n_estimators', 'max_depth', 'learning_rate', 'subsample']
        for key in expected:
            assert key in params, f"Falta '{key}' en GB params"

    def test_backward_compatibility_alias(self):
        """Verifica que get_param_distributions es alias de get_rf_param_distributions."""
        assert get_param_distributions() == get_rf_param_distributions()


class TestTuneModel:
    """Tests para la función tune_model."""

    def _make_data(self, n_samples=200):
        return make_classification(
            n_samples=n_samples, n_features=10, n_classes=2,
            n_informative=5, random_state=42,
        )

    def test_returns_correct_keys(self):
        """Verifica que el resultado contiene las claves esperadas."""
        X, y = self._make_data()
        estimator = RandomForestClassifier(random_state=42, n_jobs=1)
        params = {'n_estimators': [5, 10], 'max_depth': [3, 5]}
        result = tune_model(estimator, params, X, y, n_iter=2, cv=2, n_jobs=1)
        expected_keys = ['best_model', 'best_params', 'best_score',
                         'cv_results', 'train_test_gap']
        for key in expected_keys:
            assert key in result, f"Falta '{key}'"

    def test_best_model_is_fitted(self):
        """Verifica que el mejor modelo está entrenado."""
        X, y = self._make_data()
        estimator = RandomForestClassifier(random_state=42, n_jobs=1)
        params = {'n_estimators': [5, 10]}
        result = tune_model(estimator, params, X, y, n_iter=2, cv=2, n_jobs=1)
        predictions = result['best_model'].predict(X[:5])
        assert len(predictions) == 5

    def test_works_with_gradient_boosting(self):
        """Verifica que funciona con GradientBoosting."""
        X, y = self._make_data()
        estimator = GradientBoostingClassifier(random_state=42)
        params = {'n_estimators': [10, 20], 'max_depth': [2, 3]}
        result = tune_model(estimator, params, X, y, n_iter=2, cv=2, n_jobs=1)
        assert result['best_model'] is not None

    def test_best_score_in_valid_range(self):
        """Verifica que el mejor score está en [0, 1]."""
        X, y = self._make_data()
        estimator = RandomForestClassifier(random_state=42, n_jobs=1)
        params = {'n_estimators': [5, 10]}
        result = tune_model(estimator, params, X, y, n_iter=2, cv=2, n_jobs=1)
        assert 0 <= result['best_score'] <= 1


class TestTuneRandomForest:
    """Tests para la función de conveniencia tune_random_forest."""

    def test_returns_rf_model(self):
        """Verifica que retorna un RandomForestClassifier."""
        X, y = make_classification(n_samples=200, random_state=42)
        result = tune_random_forest(X, y, n_iter=2, cv=2, n_jobs=1)
        assert isinstance(result['best_model'], RandomForestClassifier)
