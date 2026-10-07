"""Random Forest vs ANN vs RF+ANN hybrid for tabular regression.

The hybrid is a stacked model: the ANN receives the original features plus the
Random Forest's prediction. To avoid leakage, the training-set RF predictions
are out-of-fold (cross_val_predict), so the ANN never sees a prediction the RF
made on data it was trained on.
"""

from functools import lru_cache

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import RANDOM_STATE
from app.data import get_spec, load_features


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)


def _ann() -> TransformedTargetRegressor:
    # Scale inputs and the target: length of stay is a skewed positive quantity.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=RANDOM_STATE,
            ),
        ),
        transformer=StandardScaler(),
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


@lru_cache(maxsize=4)
def run_regression_hybrid(dataset: str = "patient_length_of_stay", sample_size: int = 200) -> dict:
    spec = get_spec(dataset, expected_task="regression")
    X, y = load_features(spec)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train_v, X_test_v = X_train.values, X_test.values
    y_train_v, y_test_v = y_train.values, y_test.values

    # RF alone.
    rf = _rf().fit(X_train_v, y_train_v)
    rf_pred = rf.predict(X_test_v)

    # ANN alone.
    ann = _ann().fit(X_train_v, y_train_v)
    ann_pred = ann.predict(X_test_v)

    # RF + ANN: out-of-fold RF predictions as an extra input for the ANN.
    oof_rf = cross_val_predict(_rf(), X_train_v, y_train_v, cv=5)
    hybrid = _ann().fit(np.column_stack([X_train_v, oof_rf]), y_train_v)
    hybrid_pred = hybrid.predict(np.column_stack([X_test_v, rf_pred]))

    # A fixed, evenly spaced sample of test predictions for the scatter plot.
    idx = np.linspace(0, len(y_test_v) - 1, min(sample_size, len(y_test_v))).astype(int)
    samples = [
        {
            "actual": round(float(y_test_v[i]), 2),
            "rf": round(float(rf_pred[i]), 2),
            "ann": round(float(ann_pred[i]), 2),
            "rf_ann": round(float(hybrid_pred[i]), 2),
        }
        for i in idx
    ]

    return {
        "dataset": spec.key,
        "title": spec.title,
        "target": spec.target,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_mean_test": round(float(y_test_v.mean()), 3),
        "models": {
            "random_forest": _metrics(y_test_v, rf_pred),
            "ann": _metrics(y_test_v, ann_pred),
            "rf_ann": _metrics(y_test_v, hybrid_pred),
        },
        "samples": samples,
    }
