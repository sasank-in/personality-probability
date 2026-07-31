"""
Core prediction service: loads the model + scaler once and turns an embedding
(or, via an Embedder, raw text) into a Big Five profile.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import torch

from .config import Settings
from .model import PersonalityClassifier
from .traits import INTERPRETATIONS, TRAIT_NAMES

logger = logging.getLogger(__name__)


class PredictionError(ValueError):
    """Raised for invalid inputs to the predictor."""


class PersonalityPredictor:
    """Thread-safe for read-only inference (model in eval mode, no shared mutable state)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._device = torch.device(settings.device)

        logger.info("Loading scaler from %s", settings.scaler_path)
        self._scaler = joblib.load(settings.scaler_path)

        logger.info("Loading model from %s", settings.model_path)
        checkpoint = torch.load(
            settings.model_path, map_location=self._device, weights_only=False
        )
        self._input_dim = int(checkpoint["input_dim"])
        self._model = PersonalityClassifier(input_dim=self._input_dim).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self.test_accuracy = float(checkpoint.get("test_accuracy", float("nan")))
        self.model_version = str(checkpoint.get("model_type", "unknown"))
        logger.info(
            "Model loaded (input_dim=%d, test_accuracy=%.4f)",
            self._input_dim,
            self.test_accuracy,
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def predict_from_embedding(self, embedding: list[float] | np.ndarray) -> dict:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise PredictionError(f"Embedding must be 1-D, got shape {emb.shape}.")
        if emb.shape[0] != self._input_dim:
            raise PredictionError(
                f"Embedding must have {self._input_dim} dims, got {emb.shape[0]}."
            )
        if not np.all(np.isfinite(emb)):
            raise PredictionError("Embedding contains NaN or infinite values.")

        scaled = self._scaler.transform(emb.reshape(1, -1))
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            probs = torch.sigmoid(self._model(tensor))[0].cpu().numpy()

        preds = (probs >= 0.5).astype(int)
        return {
            "predictions": {
                t: ("High" if preds[i] else "Low") for i, t in enumerate(TRAIT_NAMES)
            },
            "probabilities": {t: float(probs[i]) for i, t in enumerate(TRAIT_NAMES)},
            "interpretation": {
                t: INTERPRETATIONS[t][int(preds[i])] for i, t in enumerate(TRAIT_NAMES)
            },
        }
