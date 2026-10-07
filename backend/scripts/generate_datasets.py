"""Generate the local datasets used by the dashboard.

The assignment asks for local datasets (Sri Lankan telecom churn, patient data).
This script builds realistic *synthetic* stand-ins with a fixed seed so the
results are reproducible. Replace the CSVs in ``backend/data`` with real data
that uses the same column names and the pipeline will pick them up unchanged.

Run from the ``backend`` folder:

    python scripts/generate_datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PROVINCES = [
    "Western", "Central", "Southern", "Northern", "Eastern",
    "North Western", "North Central", "Uva", "Sabaragamuwa",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_telecom_churn(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Prepaid", "Postpaid"], n, p=[0.7, 0.3])
    payment = rng.choice(["Mobile Wallet", "Bank Transfer", "Card", "Cash"], n, p=[0.4, 0.25, 0.2, 0.15])
    province = rng.choice(PROVINCES, n)
    roaming = rng.binomial(1, 0.15, n)
    data_gb = np.round(rng.gamma(2.0, 4.0, n), 2)
    voice_min = np.round(rng.gamma(3.0, 80.0, n), 1)
    monthly_lkr = np.round(
        300 + 40 * data_gb + 0.6 * voice_min + 350 * (contract == "Postpaid") + rng.normal(0, 150, n), 2
    ).clip(200, None)
    complaints = rng.poisson(0.6, n)
    support_calls = rng.poisson(1.2, n) + complaints

    # Churn is driven by short tenure, prepaid status, complaints and high bills.
    logit = (
        0.4
        - 0.045 * tenure
        + 0.9 * (contract == "Prepaid")
        + 0.45 * complaints
        + 0.12 * support_calls
        + 0.0025 * (monthly_lkr - 1000)
        - 0.12 * data_gb
        + 0.4 * (payment == "Cash")
        + 0.3 * roaming
    )
    churn = rng.binomial(1, sigmoid(logit))

    return pd.DataFrame(
        {
            "customer_id": [f"CUST{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract_type": contract,
            "payment_method": payment,
            "province": province,
            "monthly_charge_lkr": monthly_lkr,
            "data_usage_gb": data_gb,
            "voice_minutes": voice_min,
            "roaming_enabled": roaming,
            "complaints_last_6m": complaints,
            "support_calls_last_6m": support_calls,
            "churned": churn,
        }
    )


def make_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = rng.integers(18, 90, n)
    sex = rng.choice(["F", "M"], n)
    bmi = np.round(rng.normal(26, 5, n).clip(15, 50), 1)
    smoker = rng.binomial(1, 0.22, n)
    systolic = np.round(100 + 0.4 * age + 4 * (bmi - 25) + rng.normal(0, 12, n)).astype(int)
    glucose = np.round(85 + 0.5 * (bmi - 25) * 2 + rng.normal(0, 18, n)).clip(60, None).astype(int)
    cholesterol = np.round(160 + 0.5 * age + 2 * (bmi - 25) + rng.normal(0, 25, n)).astype(int)
    heart_rate = np.round(rng.normal(74, 9, n)).clip(45, 130).astype(int)
    prior_admissions = rng.poisson(0.7 + 0.02 * np.maximum(age - 40, 0), n)
    comorbidities = rng.poisson(0.5 + 0.03 * np.maximum(age - 40, 0), n)

    # Length of stay (days) grows with age, comorbidities and prior admissions.
    los = (
        1.5
        + 0.03 * age
        + 0.9 * comorbidities
        + 0.4 * prior_admissions
        + 0.8 * smoker
        + 0.02 * np.maximum(glucose - 110, 0)
        + rng.gamma(2.0, 0.6, n)
    )
    los = np.round(los, 1).clip(1, None)

    # 30-day readmission risk.
    logit = -2.6 + 0.25 * prior_admissions + 0.35 * comorbidities + 0.01 * (age - 60) + 0.5 * smoker + 0.15 * (los - 4)
    readmitted = rng.binomial(1, sigmoid(logit))

    return pd.DataFrame(
        {
            "patient_id": [f"P{20000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "smoker": smoker,
            "systolic_bp": systolic,
            "glucose_mg_dl": glucose,
            "cholesterol_mg_dl": cholesterol,
            "heart_rate": heart_rate,
            "prior_admissions": prior_admissions,
            "comorbidities": comorbidities,
            "length_of_stay_days": los,
            "readmitted_30d": readmitted,
        }
    )


def make_monthly_subscribers(months: int, rng: np.random.Generator) -> pd.DataFrame:
    """Monthly active subscribers (thousands): trend + yearly season + AR(1) + nonlinear shock."""
    t = np.arange(months)
    trend = 4000 + 18 * t
    season = 120 * np.sin(2 * np.pi * t / 12) + 60 * np.cos(2 * np.pi * t / 6)

    noise = np.zeros(months)
    for i in range(1, months):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 25)

    # Non-linear component (promotional spikes) that a linear ARIMA cannot capture well.
    spikes = np.zeros(months)
    for start in range(10, months, 24):
        spikes[start : start + 2] = [180, 90]
    nonlinear = 40 * np.sin(t / 3.0) * (t > 30)

    subscribers = trend + season + noise + spikes + nonlinear
    dates = pd.date_range("2015-01-01", periods=months, freq="MS")
    traffic_tb = np.round(0.9 * subscribers / 1000 * (1 + 0.004 * t) + rng.normal(0, 0.05, months), 3)
    return pd.DataFrame(
        {
            "month": dates.strftime("%Y-%m"),
            "active_subscribers_k": np.round(subscribers, 1),
            "data_traffic_tb": traffic_tb,
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    churn = make_telecom_churn(5000, rng)
    patients = make_patients(3000, rng)
    series = make_monthly_subscribers(120, rng)

    churn.to_csv(DATA_DIR / "telecom_churn.csv", index=False)
    patients.to_csv(DATA_DIR / "patient_records.csv", index=False)
    series.to_csv(DATA_DIR / "telecom_subscribers_monthly.csv", index=False)

    print(f"telecom_churn.csv               {len(churn):>6} rows  churn rate={churn['churned'].mean():.2%}")
    print(f"patient_records.csv             {len(patients):>6} rows  readmission rate={patients['readmitted_30d'].mean():.2%}")
    print(f"telecom_subscribers_monthly.csv {len(series):>6} rows")


if __name__ == "__main__":
    main()
