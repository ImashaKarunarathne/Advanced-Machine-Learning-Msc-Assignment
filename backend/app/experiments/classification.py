"""Random Forest vs XGBoost on the tabular classification datasets."""

import time
from functools import lru_cache

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.config import RANDOM_STATE
from app.data import get_spec, load_features


def build_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": X_train.columns[i], "importance": round(float(importances[i]), 4)} for i in order
    ]

    return {
        "metrics": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "train_seconds": round(train_seconds, 3),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_features": top_features,
    }


@lru_cache(maxsize=8)
def run_classification(dataset: str) -> dict:
    spec = get_spec(dataset, expected_task="classification")
    X, y = load_features(spec)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {
        name: _evaluate(model, X_train, y_train, X_test, y_test)
        for name, model in build_models().items()
    }

    rf, xgb = results["random_forest"]["metrics"], results["xgboost"]["metrics"]
    winner = "random_forest" if rf["f1"] >= xgb["f1"] else "xgboost"

    return {
        "dataset": spec.key,
        "title": spec.title,
        "target": spec.target,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "models": results,
        "best_by_f1": winner,
    }
