"""API tests using a stub embedder and the real model artifacts."""

from __future__ import annotations


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_from_text(client):
    resp = client.post(
        "/predict", json={"text": "I love meeting new people and trying new things."}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["traits"]) == 5
    for t in body["traits"]:
        assert t["prediction"] in {"High", "Low"}
        assert 0.0 <= t["probability"] <= 1.0


def test_predict_blank_text_rejected(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_predict_embedding_endpoint(client, predictor):
    vec = [0.0] * predictor.input_dim
    resp = client.post("/predict/embedding", json={"embedding": vec})
    assert resp.status_code == 200
    assert len(resp.json()["traits"]) == 5


def test_predict_embedding_wrong_dim(client, predictor):
    resp = client.post("/predict/embedding", json={"embedding": [0.0] * 10})
    assert resp.status_code == 422


def test_predict_text_too_long(client):
    resp = client.post("/predict", json={"text": "a" * 20_001})
    assert resp.status_code == 413
