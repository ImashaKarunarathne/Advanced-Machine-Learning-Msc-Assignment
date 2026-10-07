import numpy as np
import pandas as pd
import pytest

from app.config import DATA_DIR
from app.data import DATASETS, get_spec, load_features
from app.experiments.classification import run_classification
from app.experiments.regression import run_regression_hybrid
from app.experiments.timeseries import TEST_MONTHS, _hybrid_forecast, run_timeseries_hybrid


def test_every_dataset_file_exists():
    for spec in DATASETS.values():
        assert (DATA_DIR / spec.file).exists(), spec.file


def test_features_exclude_target_and_ids():
    X, y = load_features(get_spec("telecom_churn", "classification"))
    assert "churned" not in X.columns
    assert "customer_id" not in X.columns
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}


def test_wrong_task_is_rejected():
    with pytest.raises(ValueError):
        get_spec("telecom_churn", expected_task="regression")


def test_classification_reports_both_models():
    result = run_classification("telecom_churn")
    assert set(result["models"]) == {"random_forest", "xgboost"}
    for model in result["models"].values():
        assert 0.5 < model["metrics"]["roc_auc"] <= 1.0
        cm = model["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == result["test_rows"]
    assert result["best_by_f1"] in result["models"]


def test_timeseries_hybrid_beats_arima_alone():
    result = run_timeseries_hybrid()
    assert len(result["series"]) == TEST_MONTHS
    models = result["models"]
    # The RF residual correction should improve on the ARIMA baseline it builds on.
    assert models["rf_arima"]["mae"] < models["arima"]["mae"]


def test_timeseries_forecasts_use_only_past_data():
    """A one-step forecast for month t must not change when only month t (or later) changes."""
    df = pd.read_csv(DATA_DIR / "telecom_subscribers_monthly.csv")
    months = pd.to_datetime(df["month"] + "-01")
    y = df["active_subscribers_k"].astype(float).values
    split = len(y) - TEST_MONTHS

    base = _hybrid_forecast(y, months, split)
    perturbed = y.copy()
    k = split + 5  # the 6th test month
    perturbed[k] += 10_000
    changed = _hybrid_forecast(perturbed, months, split)

    # Test positions 0..5 are months split..k, all forecast from data before k: unchanged.
    np.testing.assert_allclose(base["arima_pred"][:6], changed["arima_pred"][:6], rtol=1e-8)
    np.testing.assert_allclose(base["hybrid_pred"][:6], changed["hybrid_pred"][:6], rtol=1e-8)
    # Later months do depend on the perturbed value, so the test is not trivially passing.
    assert not np.allclose(base["arima_pred"][6:], changed["arima_pred"][6:])


def test_regression_hybrid_reports_three_models():
    result = run_regression_hybrid()
    assert set(result["models"]) == {"random_forest", "ann", "rf_ann"}
    for model in result["models"].values():
        assert model["r2"] > 0.5
    assert len(result["samples"]) <= 200
