"""Pydantic schemas for the FastAPI inference service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for code-risk classification."""

    code: str = Field(..., description="Source-code snippet to classify.")


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    label: Literal["clean", "buggy"]
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
