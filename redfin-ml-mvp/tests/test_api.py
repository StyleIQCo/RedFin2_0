"""End-to-end API test — exercises the whole train → register → serve path.

This is the single most important integration test: it proves the artifact
written by training matches what serving loads. In a real CI pipeline,
this runs on every commit.
"""
from fastapi.testclient import TestClient

from data.generate import generate
from src.avm.train import main as train_avm
from src.recommender.train import main as train_rec
from src.config import settings


def _setup_everything(tmp_path, monkeypatch):
    """Redirect all state to tmp_path and run the full train pipeline."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "registry_dir", tmp_path / "registry" / "models")
    monkeypatch.setattr(settings, "artifacts_dir", tmp_path / "artifacts")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "registry" / "models").mkdir(parents=True, exist_ok=True)
    generate(n=3_000, out=tmp_path / "data" / "homes.parquet")
    train_avm(tmp_path / "data" / "homes.parquet")
    train_rec(tmp_path / "data" / "homes.parquet")


def test_full_pipeline_api(tmp_path, monkeypatch):
    _setup_everything(tmp_path, monkeypatch)

    # Import AFTER monkeypatching so the app uses redirected settings
    from src.serving import app as app_module
    client = TestClient(app_module.app)

    # Healthz
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Predict
    payload = {
        "city": "Seattle", "property_type": "single_family",
        "sqft": 2000, "beds": 3, "baths": 2.0, "lot_size": 5000,
        "year_built": 2000, "garage_spaces": 2,
        "school_score": 7.0, "walk_score": 75, "crime_index": 35,
    }
    r = client.post("/v1/avm/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["point"] > 0
    assert body["lower"] <= body["point"] <= body["upper"]
    assert body["model_name"] == settings.avm_model_name

    # Batch
    r = client.post("/v1/avm/batch", json={"homes": [payload, payload]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2

    # Recommender
    r = client.post("/v1/recommender/similar", json={"listing_id": 1, "k": 5, "same_city": True})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) >= 1

    # Metrics endpoint returns Prometheus text
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "redfin_ml_requests_total" in r.text
