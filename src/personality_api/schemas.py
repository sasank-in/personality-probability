"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .traits import TRAIT_NAMES


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Free text to analyze.")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class EmbeddingRequest(BaseModel):
    embedding: list[float] = Field(
        ..., description="Pre-computed embedding vector (must match model input_dim)."
    )


class TraitResult(BaseModel):
    trait: str
    prediction: str  # "High" | "Low"
    probability: float
    interpretation: str


class PredictionResponse(BaseModel):
    traits: list[TraitResult]
    model_version: str

    @classmethod
    def from_profile(cls, profile: dict, model_version: str) -> PredictionResponse:
        traits = [
            TraitResult(
                trait=t,
                prediction=profile["predictions"][t],
                probability=profile["probabilities"][t],
                interpretation=profile["interpretation"][t],
            )
            for t in TRAIT_NAMES
        ]
        return cls(traits=traits, model_version=model_version)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    embedder: str
    version: str
