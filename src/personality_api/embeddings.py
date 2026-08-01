"""
Pluggable text -> embedding providers.

Predictions are only valid on embeddings from the SAME model the classifier was
trained on. The shipped model was trained on local `sentence-transformers`
embeddings (see the checkpoint's `embed_model` / `embed_prefix` /
`normalize_embeddings` fields), so LocalEmbedder is the default production path
and reads those settings straight from the checkpoint to guarantee the service
embeds exactly how the model was trained.

- LocalEmbedder  : offline sentence-transformers model (default).
- MistralEmbedder: Mistral hosted API (only if the model was trained on it).
- RawEmbedder    : passthrough for /predict/embedding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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


class LocalEmbedder(Embedder):
    """
    Offline sentence-transformers embedder. Model name, instruction prefix and
    normalization are read from the checkpoint so inference matches training.
    """

    def __init__(self, settings: Settings, checkpoint: dict[str, Any] | None = None):
        checkpoint = checkpoint or {}
        # Checkpoint is the source of truth; settings only override if the
        # checkpoint lacks the field (older artifacts).
        self._model_name = checkpoint.get("embed_model") or settings.local_embed_model
        self._prefix = checkpoint.get("embed_prefix", settings.local_embed_prefix)
        self._normalize = bool(checkpoint.get("normalize_embeddings", settings.local_normalize))
        self._dim = int(checkpoint.get("embed_dim") or settings.embedding_dim)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise EmbeddingError(
                "The 'sentence-transformers' package is required for the local embedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        try:
            self._encoder = SentenceTransformer(self._model_name, device=settings.device)
        except Exception as exc:  # noqa: BLE001 - model download/load failures
            raise EmbeddingError(
                f"Failed to load local embedding model '{self._model_name}': {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        try:
            vec = self._encoder.encode(
                [self._prefix + text],
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc

        vector = vec.astype(float).tolist()
        if len(vector) != self._dim:
            raise EmbeddingError(
                f"Expected {self._dim}-dim embedding, got {len(vector)} from {self._model_name}."
            )
        return vector

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


class MistralEmbedder(Embedder):
    """Calls Mistral's hosted embeddings API (only valid if the model used it)."""

    def __init__(self, settings: Settings):
        if not settings.mistral_api_key:
            raise EmbeddingError(
                "PERSONALITY_MISTRAL_API_KEY is not set; required for the 'mistral' embedder."
            )
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
            "with a pre-computed vector, or configure a text embedder."
        )

    @property
    def name(self) -> str:
        return "raw"


def build_embedder(settings: Settings, checkpoint: dict[str, Any] | None = None) -> Embedder:
    """Factory selecting the embedder from settings (checkpoint informs LocalEmbedder)."""
    kind = settings.embedder.lower()
    if kind == "local":
        return LocalEmbedder(settings, checkpoint)
    if kind == "mistral":
        return MistralEmbedder(settings)
    if kind == "raw":
        return RawEmbedder()
    raise EmbeddingError(
        f"Unknown embedder '{settings.embedder}' (expected 'local', 'mistral', or 'raw')."
    )
