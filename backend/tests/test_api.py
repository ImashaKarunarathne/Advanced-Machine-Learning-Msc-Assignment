from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_lists_all_datasets():
    body = client.get("/api/datasets").json()
    keys = {d["key"] for d in body}
    assert keys == {"telecom_churn", "patient_readmission", "patient_length_of_stay", "subscribers_monthly"}
    churn = next(d for d in body if d["key"] == "telecom_churn")
    assert churn["rows"] == 5000
    assert churn["missing_values"] == 0


def test_classification_endpoint():
    body = client.get("/api/classification/telecom_churn").json()
    assert body["best_by_f1"] in {"random_forest", "xgboost"}
    assert "random_forest" in body["models"]


def test_unknown_dataset_is_404():
    assert client.get("/api/classification/nope").status_code == 404


def test_regression_dataset_on_classification_route_is_400():
    assert client.get("/api/classification/patient_length_of_stay").status_code == 400


def test_timeseries_endpoint():
    body = client.get("/api/hybrid/timeseries").json()
    assert set(body["models"]) == {"arima", "random_forest", "rf_arima"}
    assert len(body["series"]) == body["test_months"]


def test_regression_endpoint():
    body = client.get("/api/hybrid/regression").json()
    assert set(body["models"]) == {"random_forest", "ann", "rf_ann"}
