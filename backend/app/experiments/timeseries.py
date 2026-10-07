"""Hybrid RF + ARIMA forecasting on the monthly subscriber series.

Pipeline (all forecasts are one-step-ahead, so no future values leak in):

1. ARIMA is fitted on the training period only.
2. The fitted ARIMA parameters are applied to the full series, giving the
   one-step-ahead ARIMA prediction for every month.
3. A Random Forest learns the ARIMA residuals from lagged residuals and
   calendar features. Hybrid forecast = ARIMA prediction + RF residual forecast.

An RF-only baseline (lagged values of the series) and an ARIMA-only baseline
are reported alongside the hybrid so the comparison is like for like.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

from app.config import RANDOM_STATE
from app.data import get_spec, load_frame

ARIMA_ORDER = (2, 1, 2)
TEST_MONTHS = 24
RESIDUAL_LAGS = 6
VALUE_LAGS = 12


def _calendar_features(months) -> pd.DataFrame:
    month_index = pd.DatetimeIndex(months)
    m = month_index.month.values
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * m / 12),
            "month_cos": np.cos(2 * np.pi * m / 12),
        },
        index=range(len(month_index)),
    )


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 3),
        "mape": round(float(np.mean(np.abs(err / actual)) * 100), 3),
    }


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)


def _rf_only_forecast(y: np.ndarray, months: pd.DatetimeIndex, split: int) -> np.ndarray:
    """RF on lagged values (plus calendar features), walk-forward over the test window."""
    cal = _calendar_features(months).values
    rows, targets = [], []
    for t in range(VALUE_LAGS, len(y)):
        rows.append(np.concatenate([y[t - VALUE_LAGS : t][::-1], cal[t]]))
        targets.append(y[t])
    X, target = np.array(rows), np.array(targets)
    # Row k corresponds to time index k + VALUE_LAGS.
    train_mask = np.arange(len(target)) + VALUE_LAGS < split
    model = _rf().fit(X[train_mask], target[train_mask])
    return model.predict(X[~train_mask])


def _hybrid_forecast(y: np.ndarray, months: pd.DatetimeIndex, split: int) -> dict:
    # Step 1-2: ARIMA fitted on the training period, then filtered over the full series.
    base = ARIMA(y[:split], order=ARIMA_ORDER).fit()
    one_step = ARIMA(y, order=ARIMA_ORDER).filter(base.params)
    arima_pred = np.asarray(one_step.fittedvalues)
    residual = y - arima_pred

    # Step 3: RF learns the residual from past residuals, past values and calendar features.
    cal = _calendar_features(months).values
    rows, targets = [], []
    for t in range(RESIDUAL_LAGS, len(y)):
        rows.append(
            np.concatenate([residual[t - RESIDUAL_LAGS : t][::-1], [arima_pred[t]], cal[t]])
        )
        targets.append(residual[t])
    X, target = np.array(rows), np.array(targets)
    idx = np.arange(len(target)) + RESIDUAL_LAGS
    train_mask = idx < split

    rf = _rf().fit(X[train_mask], target[train_mask])
    residual_pred = rf.predict(X[~train_mask])
    test_idx = idx[~train_mask]
    hybrid_pred = arima_pred[test_idx] + residual_pred

    return {
        "arima_order": list(ARIMA_ORDER),
        "arima_pred": arima_pred[test_idx],
        "hybrid_pred": hybrid_pred,
        "test_idx": test_idx,
        "arima_aic": round(float(base.aic), 2),
    }


@lru_cache(maxsize=2)
def run_timeseries_hybrid(dataset: str = "subscribers_monthly", test_months: int = TEST_MONTHS) -> dict:
    spec = get_spec(dataset, expected_task="timeseries")
    df = load_frame(spec)
    months = pd.to_datetime(df["month"] + "-01")
    y = df[spec.target].astype(float).values
    split = len(y) - test_months

    hybrid = _hybrid_forecast(y, months, split)
    rf_only = _rf_only_forecast(y, months, split)

    # Every model is scored on the same months [split, n). Each sequence has exactly
    # test_months entries because the walk-forward windows end at the last observation.
    test_actual = y[split:]
    arima_pred = hybrid["arima_pred"]
    hybrid_pred = hybrid["hybrid_pred"]
    rf_pred = rf_only

    series = [
        {
            "month": df["month"].iloc[split + i],
            "actual": round(float(test_actual[i]), 2),
            "arima": round(float(arima_pred[i]), 2),
            "rf": round(float(rf_pred[i]), 2),
            "rf_arima": round(float(hybrid_pred[i]), 2),
        }
        for i in range(test_months)
    ]

    return {
        "dataset": spec.key,
        "title": spec.title,
        "target": spec.target,
        "train_months": int(split),
        "test_months": int(test_months),
        "arima_order": hybrid["arima_order"],
        "arima_aic": hybrid["arima_aic"],
        "models": {
            "arima": _metrics(test_actual, arima_pred),
            "random_forest": _metrics(test_actual, rf_pred),
            "rf_arima": _metrics(test_actual, hybrid_pred),
        },
        "series": series,
    }
