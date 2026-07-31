# Big Five Personality API

Predict Big Five (OCEAN) personality traits from text, served as a production
FastAPI service.

- **Input:** free text → embedded via Mistral `mistral-embed` (1024-dim).
- **Model:** a regularized linear classifier (5 L2-penalized logistic
  regressions). On this dataset (~1.6k essays) it beats a deep MLP on the
  held-out test set (**62.0%** mean accuracy vs 58.4%) without overfitting.
- **Output:** per-trait High/Low prediction, probability, and interpretation.

> **Realistic expectation:** predicting Big Five from text tops out around
> 60–63% in the literature. This model sits near that ceiling; a bigger model
> won't help — only more/richer data would.

## Why Mistral embeddings

The model was trained on `mistral-embed` vectors, so predictions are only valid
on embeddings from the same provider. Swapping in a different embedder (e.g.
`sentence-transformers`) without retraining would silently destroy accuracy.
The embedder is pluggable (`src/personality_api/embeddings.py`) if you ever
retrain on a different provider.

## Quickstart

```bash
# 1. Install (CPU torch shown; use the CUDA wheel if you have a GPU)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[mistral,dev]"

# 2. Configure
cp .env.example .env
# set PERSONALITY_MISTRAL_API_KEY=... in .env

# 3. Run
personality-serve            # -> http://localhost:8000  (docs at /docs)
```

## API

| Method | Path                 | Body                              | Notes                              |
|--------|----------------------|-----------------------------------|------------------------------------|
| GET    | `/healthz`           | —                                 | Liveness + model/embedder status   |
| POST   | `/predict`           | `{"text": "..."}`                 | Embeds text, then predicts         |
| POST   | `/predict/embedding` | `{"embedding": [f1, ..., f1024]}` | Skip embedding; supply your own    |

Interactive docs at `/docs` (Swagger) and `/redoc`.

### Example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I love meeting new people and exploring bold new ideas."}'
```

```json
{
  "traits": [
    {"trait": "Openness", "prediction": "High", "probability": 0.71,
     "interpretation": "Creative, open to new experiences"},
    "..."
  ],
  "model_version": "logistic_regression_linear"
}
```

To try it **without** a Mistral key, use `/predict/embedding` with a 1024-dim
vector (e.g. from the training dataset).

## Configuration

All settings are environment variables prefixed `PERSONALITY_` (see
[`.env.example`](.env.example)). Key ones:

| Variable                       | Default        | Purpose                              |
|--------------------------------|----------------|--------------------------------------|
| `PERSONALITY_EMBEDDER`         | `mistral`      | `mistral` or `raw`                   |
| `PERSONALITY_MISTRAL_API_KEY`  | —              | Required for the `mistral` embedder  |
| `PERSONALITY_DEVICE`           | `cpu`          | `cpu` or `cuda`                      |
| `PERSONALITY_MODEL_PATH`       | `artifacts/…`  | Model checkpoint location            |

## Docker

```bash
docker compose up --build
# or
docker build -t personality-api .
docker run -p 8000:8000 -e PERSONALITY_MISTRAL_API_KEY=sk-... personality-api
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests     # lint
pytest -q                # tests run offline (stub embedder + shipped artifacts)
```

CI (`.github/workflows/ci.yml`) runs lint + tests on Python 3.10 and 3.12.

## Retraining

```bash
pip install -e ".[train]"
python scripts/train.py   # writes artifacts/personality_model.pt + x_scaler.pkl
```

## Project layout

```
src/personality_api/
  api.py            FastAPI app (/predict, /predict/embedding, /healthz)
  predictor.py      loads model+scaler once; embedding -> profile
  embeddings.py     pluggable embedders (Mistral, raw)
  model.py          PersonalityClassifier (linear)
  schemas.py        request/response models
  config.py         env-driven settings
  traits.py         trait names + interpretations
artifacts/          personality_model.pt, x_scaler.pkl
scripts/train.py    training / retraining
tests/              pytest suite (offline)
```
