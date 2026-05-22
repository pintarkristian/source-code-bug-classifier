"""FastAPI application for code bug/risk classification."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response, status

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import get_settings
from src.predict import CodeBugPredictor

LOGGER = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="Classify source-code snippets as clean or potentially buggy/risky.",
)


@lru_cache(maxsize=1)
def get_predictor() -> CodeBugPredictor:
    """Return the cached predictor used by the API.

    The predictor is created lazily so tests can override this dependency without
    loading a real Transformer model.
    """
    return CodeBugPredictor(
        model_dir=settings.model_dir,
        model_name=settings.default_model_name,
    )


PredictorDependency = Annotated[CodeBugPredictor, Depends(get_predictor)]


@app.get("/health", response_model=HealthResponse)
def health(predictor: PredictorDependency, response: Response) -> HealthResponse:
    """Return service health and whether a trained model was loaded."""
    model_loaded = bool(getattr(predictor, "is_loaded", False))
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: PredictorDependency,
) -> PredictionResponse:
    """Classify a source-code snippet.

    Empty or whitespace-only code is rejected with HTTP 400 because the request
    shape is valid JSON, but the snippet content is not useful for inference.
    """
    if not request.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="code must not be empty",
        )

    try:
        result = predictor.predict(request.code)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        LOGGER.warning("Prediction service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="prediction service unavailable",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected prediction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="prediction failed",
        ) from exc

    return PredictionResponse(**result)
