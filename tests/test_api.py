"""Tests for the FastAPI app."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_predictor


class DummyPredictor:
    """Small test double that avoids loading a real Transformer model."""

    def __init__(self, is_loaded: bool = True) -> None:
        self.is_loaded = is_loaded

    def predict(self, code: str) -> dict:
        """Return a deterministic API-compatible prediction payload."""
        notes: list[str] = []
        risk_score = 0.82 if "/" in code else 0.12
        label = "buggy" if risk_score >= 0.5 else "clean"

        if "/" in code:
            notes.append("Potential division-by-zero risk")

        return {
            "label": label,
            "risk_score": risk_score,
            "model_name": "test-codebert",
            "notes": notes,
        }


class FailingPredictor(DummyPredictor):
    """Test double that raises a configured exception during prediction."""

    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception

    def predict(self, code: str) -> dict:
        """Raise the configured exception instead of returning a prediction."""
        raise self.exception


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Create a test client with the predictor dependency mocked."""
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_reports_ok_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_endpoint_reports_degraded_when_model_missing() -> None:
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor(is_loaded=False)
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_predict_endpoint_accepts_valid_code(client: TestClient) -> None:
    response = client.post("/predict", json={"code": "def divide(a, b): return a / b"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "label": "buggy",
        "risk_score": 0.82,
        "model_name": "test-codebert",
        "notes": ["Potential division-by-zero risk"],
    }


def test_predict_endpoint_rejects_blank_code(client: TestClient) -> None:
    response = client.post("/predict", json={"code": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "code must not be empty"


def test_predict_endpoint_returns_bad_request_for_predictor_value_error() -> None:
    app.dependency_overrides[get_predictor] = lambda: FailingPredictor(ValueError("invalid code"))
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"code": "def broken(): pass"})
    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid code"


def test_predict_endpoint_returns_unavailable_for_predictor_runtime_error() -> None:
    app.dependency_overrides[get_predictor] = lambda: FailingPredictor(RuntimeError("model unavailable"))
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"code": "def broken(): pass"})
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "prediction service unavailable"


def test_predict_endpoint_returns_controlled_error_for_unexpected_exception() -> None:
    app.dependency_overrides[get_predictor] = lambda: FailingPredictor(Exception("boom"))
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"code": "def broken(): pass"})
    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == "prediction failed"
