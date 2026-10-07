# Random Forest Dashboard

Hands-on project for Applied ML: a Random Forest dashboard that compares RF with XGBoost on local
datasets and tries two hybrids (RF + ARIMA for time series, RF + ANN for tabular regression).

```
backend/    FastAPI service, experiments, tests, local CSV datasets
frontend/   Vite + React dashboard
```

## Datasets

The assignment asks for local data (Sri Lankan telecom churn, patient records). The CSVs in
`backend/data/` are **synthetic**, generated with a fixed seed by `backend/scripts/generate_datasets.py`,
so results are reproducible. To use real data, replace the files with CSVs that have the same column
names (see `backend/app/data.py` for the registry).

| Dataset | File | Task | Target |
|---|---|---|---|
| Telecom churn | `telecom_churn.csv` | classification | `churned` |
| Patient readmission | `patient_records.csv` | classification | `readmitted_30d` |
| Patient length of stay | `patient_records.csv` | regression | `length_of_stay_days` |
| Monthly subscribers | `telecom_subscribers_monthly.csv` | time series | `active_subscribers_k` |

## Models

- **Random Forest vs XGBoost**: stratified 80/20 split; accuracy, precision, recall, F1, ROC-AUC, training time, confusion matrix, feature importance.
- **RF + ARIMA**: ARIMA(2,1,2) gives one-step-ahead forecasts; a random forest models its residuals from lagged residuals and calendar features. Compared with ARIMA alone and RF alone.
- **RF + ANN**: an MLP regressor with a scaled target, plus a stacked version that receives out-of-fold RF predictions. Compared with RF alone and ANN alone.

## Running locally

Backend (Python 3.12):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/generate_datasets.py   # only needed once
uvicorn app.main:app --reload --port 8000
```

Frontend (Node 18+), in a second terminal:

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173, /api is proxied to port 8000
```

Tests:

```bash
cd backend
pytest
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/datasets` | Dataset summaries |
| GET | `/api/classification/{telecom_churn\|patient_readmission}` | RF vs XGBoost results |
| GET | `/api/hybrid/timeseries` | ARIMA, RF and RF+ARIMA forecasts and errors |
| GET | `/api/hybrid/regression` | RF, ANN and RF+ANN errors and prediction samples |

Results are computed on first request and cached in memory for the life of the process.

## Not yet covered

- Comparison with LightGBM and CatBoost (the "research" item on the slide).
- Results on real Sri Lankan telecom and patient data; the current runs use the synthetic CSVs.
