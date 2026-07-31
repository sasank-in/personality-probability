"""Entry point: `personality-serve` / `python -m personality_api`."""

from __future__ import annotations

import os

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "personality_api.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
