"""Shared test fixtures. Uses the real shipped artifacts; no network required."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from personality_api.config import get_settings
from personality_api.embeddings import Embedder
from personality_api.predictor import PersonalityPredictor


class StubEmbedder(Embedder):
    """Deterministic fake embedder — returns a fixed-dim zero-ish vector."""

    def __init__(self, dim: int):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        # Deterministic, finite, right shape — enough to exercise the pipeline.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self._dim).astype(float).tolist()

    @property
    def name(self) -> str:
        return "stub"


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def predictor(settings) -> PersonalityPredictor:
    return PersonalityPredictor(settings)


@pytest.fixture
def client(predictor):
    """TestClient with model loaded and a stub embedder injected (no lifespan network)."""
    from personality_api.api import app

    app.state.settings = get_settings()
    app.state.predictor = predictor
    app.state.embedder = StubEmbedder(predictor.input_dim)
    with TestClient(app) as c:
        yield c
