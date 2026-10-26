from fastapi.testclient import TestClient

from customer_ml.api import create_app


def test_health_endpoint_does_not_require_database():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_model_returns_service_unavailable(tmp_path):
    client = TestClient(create_app(model_dir=tmp_path))

    response = client.post("/predictions/purchase/synthetic-000001")

    assert response.status_code == 503
    assert "model is unavailable" in response.json()["detail"]


def test_unknown_prediction_target_is_rejected(tmp_path):
    client = TestClient(create_app(model_dir=tmp_path))

    response = client.post("/predictions/other/synthetic-000001")

    assert response.status_code == 404
