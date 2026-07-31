"""FastAPI application exposing the Big Five personality classifier."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from . import __version__
from .config import Settings, get_settings
from .embeddings import Embedder, EmbeddingError, build_embedder
from .logging_config import configure_logging
from .predictor import PersonalityPredictor, PredictionError
from .schemas import (
    EmbeddingRequest,
    HealthResponse,
    PredictionResponse,
    TextRequest,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting personality-api v%s", __version__)

    # Idempotent: tests may pre-populate app.state to inject stubs.
    if getattr(app.state, "predictor", None) is not None:
        logger.info("Predictor already initialized; skipping startup load.")
        yield
        return

    app.state.settings = settings
    app.state.predictor = PersonalityPredictor(settings)
    # Embedder is optional at startup: the /predict/embedding path never needs it,
    # and /predict surfaces a clear 503 if the embedder is unavailable.
    try:
        app.state.embedder = build_embedder(settings)
        logger.info("Embedder ready: %s", app.state.embedder.name)
    except EmbeddingError as exc:
        app.state.embedder = None
        logger.warning("Embedder unavailable at startup: %s", exc)

    yield
    logger.info("Shutting down personality-api")


app = FastAPI(
    title="Big Five Personality API",
    version=__version__,
    description="Predict Big Five (OCEAN) personality traits from text.",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> PersonalityPredictor:
    return request.app.state.predictor


def get_embedder(request: Request) -> Embedder:
    embedder = request.app.state.embedder
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Text embedding is not configured. Set PERSONALITY_MISTRAL_API_KEY "
            "or use POST /predict/embedding with a pre-computed vector.",
        )
    return embedder


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


@app.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    embedder = getattr(request.app.state, "embedder", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        embedder=embedder.name if embedder is not None else "none",
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: TextRequest,
    predictor: PersonalityPredictor = Depends(get_predictor),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_app_settings),
) -> PredictionResponse:
    if len(req.text) > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds max length of {settings.max_text_chars} characters.",
        )
    try:
        embedding = embedder.embed(req.text)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        profile = predictor.predict_from_embedding(embedding)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse.from_profile(profile, predictor.model_version)


@app.post("/predict/embedding", response_model=PredictionResponse)
def predict_embedding(
    req: EmbeddingRequest,
    predictor: PersonalityPredictor = Depends(get_predictor),
) -> PredictionResponse:
    try:
        profile = predictor.predict_from_embedding(req.embedding)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse.from_profile(profile, predictor.model_version)
