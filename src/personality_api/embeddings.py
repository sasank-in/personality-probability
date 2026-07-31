"""
Pluggable text -> embedding providers.

The model was trained on Mistral `mistral-embed` vectors, so predictions are
only valid on embeddings from the SAME provider. MistralEmbedder is the
production path. RawEmbedder is a passthrough for testing / power users who
already hold a valid vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce a vector."""


class Embedder(ABC):
    """Turns raw text into a fixed-dim embedding vector."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MistralEmbedder(Embedder):
    """Calls Mistral's hosted embeddings API (matches the training distribution)."""

    def __init__(self, settings: Settings):
        if not settings.mistral_api_key:
            raise EmbeddingError(
                "PERSONALITY_MISTRAL_API_KEY is not set; required for the 'mistral' embedder."
            )
        # Imported lazily so the package installs/tests without the SDK present.
        try:
            from mistralai import Mistral
        except ImportError as exc:  # pragma: no cover - import guard
            raise EmbeddingError(
                "The 'mistralai' package is required for the Mistral embedder. "
                "Install it with: pip install mistralai"
            ) from exc

        self._client = Mistral(api_key=settings.mistral_api_key)
        self._model = settings.mistral_model
        self._dim = settings.embedding_dim

    def embed(self, text: str) -> list[float]:
        try:
            resp = self._client.embeddings.create(model=self._model, inputs=[text])
        except Exception as exc:  # noqa: BLE001 - surface any SDK/network failure uniformly
            raise EmbeddingError(f"Mistral embedding request failed: {exc}") from exc

        vector = list(resp.data[0].embedding)
        if len(vector) != self._dim:
            raise EmbeddingError(
                f"Expected {self._dim}-dim embedding, got {len(vector)} from {self._model}."
            )
        return vector

    @property
    def name(self) -> str:
        return f"mistral:{self._model}"


class RawEmbedder(Embedder):
    """Passthrough — `text` is ignored; callers use /predict/embedding instead."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - not a real path
        raise EmbeddingError(
            "The 'raw' embedder cannot embed text. Use the /predict/embedding endpoint "
            "with a pre-computed vector, or configure the 'mistral' embedder."
        )

    @property
    def name(self) -> str:
        return "raw"


def build_embedder(settings: Settings) -> Embedder:
    """Factory selecting the embedder from settings."""
    kind = settings.embedder.lower()
    if kind == "mistral":
        return MistralEmbedder(settings)
    if kind == "raw":
        return RawEmbedder()
    raise EmbeddingError(f"Unknown embedder '{settings.embedder}' (expected 'mistral' or 'raw').")
