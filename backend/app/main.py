"""FastAPI backend for the Random Forest dashboard."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.data import DATASETS, dataset_summary
from app.experiments.classification import run_classification
from app.experiments.regression import run_regression_hybrid
from app.experiments.timeseries import run_timeseries_hybrid

app = FastAPI(title="RF Dashboard API", version="1.0.0")

# The Vite dev server runs on 5173 and proxies /api, so CORS is only needed for direct calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    return [dataset_summary(spec) for spec in DATASETS.values()]


@app.get("/api/classification/{dataset}")
def classification(dataset: str) -> dict:
    try:
        return run_classification(dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/hybrid/timeseries")
def hybrid_timeseries() -> dict:
    return run_timeseries_hybrid()


@app.get("/api/hybrid/regression")
def hybrid_regression() -> dict:
    return run_regression_hybrid()
