"""Environment-driven configuration. No hardcoded paths or secrets in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/personality_api/config.py).
_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PERSONALITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Artifact locations (overridable via env for containers / cloud storage).
    model_path: Path = Field(default=_ROOT / "artifacts" / "personality_model.pt")
    scaler_path: Path = Field(default=_ROOT / "artifacts" / "x_scaler.pkl")

    # Embedding provider: "mistral" (real) or "raw" (caller supplies vector).
    embedder: str = Field(default="mistral")
    mistral_api_key: str | None = Field(default=None)
    mistral_model: str = Field(default="mistral-embed")
    embedding_dim: int = Field(default=1024)

    # Input guardrails.
    max_text_chars: int = Field(default=20_000)

    # Runtime.
    device: str = Field(default="cpu")  # "cpu" | "cuda"
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    """Load settings once (module-level singleton via lru_cache-like pattern)."""
    return _settings


_settings = Settings()
