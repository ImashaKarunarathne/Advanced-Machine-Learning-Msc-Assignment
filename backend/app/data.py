"""Dataset registry and loaders for the local CSV files."""

from dataclasses import dataclass, field

import pandas as pd

from app.config import DATA_DIR


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    file: str
    task: str  # "classification" | "regression" | "timeseries"
    target: str
    drop: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in [
        DatasetSpec(
            key="telecom_churn",
            title="Telecom churn (Sri Lanka)",
            file="telecom_churn.csv",
            task="classification",
            target="churned",
            drop=("customer_id",),
            description="Customer-level churn for a mobile operator: tenure, billing, usage and complaints.",
        ),
        DatasetSpec(
            key="patient_readmission",
            title="Patient 30-day readmission",
            file="patient_records.csv",
            task="classification",
            target="readmitted_30d",
            drop=("patient_id", "length_of_stay_days"),
            description="Hospital patient records; predicts whether a patient is readmitted within 30 days.",
        ),
        DatasetSpec(
            key="patient_length_of_stay",
            title="Patient length of stay",
            file="patient_records.csv",
            task="regression",
            target="length_of_stay_days",
            drop=("patient_id", "readmitted_30d"),
            description="Hospital patient records; predicts length of stay in days.",
        ),
        DatasetSpec(
            key="subscribers_monthly",
            title="Monthly active subscribers",
            file="telecom_subscribers_monthly.csv",
            task="timeseries",
            target="active_subscribers_k",
            description="Monthly active subscribers (thousands), 2015 to 2024.",
        ),
    ]
}


def get_spec(key: str, expected_task: str | None = None) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'")
    spec = DATASETS[key]
    if expected_task and spec.task != expected_task:
        raise ValueError(f"Dataset '{key}' is a {spec.task} dataset, not {expected_task}")
    return spec


def load_frame(spec: DatasetSpec) -> pd.DataFrame:
    path = DATA_DIR / spec.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/generate_datasets.py` first.")
    return pd.read_csv(path)


def load_features(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Return one-hot encoded features and the target for a tabular dataset."""
    df = load_frame(spec)
    y = df[spec.target]
    X = df.drop(columns=[spec.target, *spec.drop])
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return X, y


def dataset_summary(spec: DatasetSpec) -> dict:
    df = load_frame(spec)
    summary = {
        "key": spec.key,
        "title": spec.title,
        "task": spec.task,
        "target": spec.target,
        "description": spec.description,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
    }
    if spec.task == "classification":
        summary["positive_rate"] = round(float(df[spec.target].mean()), 4)
    else:
        summary["target_mean"] = round(float(df[spec.target].mean()), 3)
    return summary
