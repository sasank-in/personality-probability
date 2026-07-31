"""Unit tests for the core predictor (no HTTP layer)."""

from __future__ import annotations

import numpy as np
import pytest

from personality_api.predictor import PredictionError
from personality_api.traits import TRAIT_NAMES


def test_predict_shape_and_keys(predictor):
    emb = np.zeros(predictor.input_dim, dtype=np.float32)
    result = predictor.predict_from_embedding(emb)
    assert set(result["predictions"]) == set(TRAIT_NAMES)
    assert set(result["probabilities"]) == set(TRAIT_NAMES)
    for t in TRAIT_NAMES:
        assert result["predictions"][t] in {"High", "Low"}
        assert 0.0 <= result["probabilities"][t] <= 1.0
        assert isinstance(result["interpretation"][t], str)


def test_predict_wrong_dim_raises(predictor):
    with pytest.raises(PredictionError):
        predictor.predict_from_embedding([0.0] * (predictor.input_dim - 1))


def test_predict_rejects_nan(predictor):
    emb = np.zeros(predictor.input_dim, dtype=np.float32)
    emb[0] = np.nan
    with pytest.raises(PredictionError):
        predictor.predict_from_embedding(emb)


def test_predict_rejects_2d(predictor):
    with pytest.raises(PredictionError):
        predictor.predict_from_embedding(np.zeros((2, predictor.input_dim)))


def test_deterministic(predictor):
    emb = np.linspace(-1, 1, predictor.input_dim, dtype=np.float32)
    r1 = predictor.predict_from_embedding(emb)
    r2 = predictor.predict_from_embedding(emb)
    assert r1["probabilities"] == r2["probabilities"]
