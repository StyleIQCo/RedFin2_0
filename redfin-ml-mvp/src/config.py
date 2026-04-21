"""Central config — all paths, model names, thresholds in one place.

In production this would be powered by a config service (Consul, AWS AppConfig,
or env vars bundled via a typed settings class). Single source of truth so we
never hard-code paths across modules.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    repo_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = repo_root / "data"
    registry_dir: Path = repo_root / "registry" / "models"
    artifacts_dir: Path = repo_root / "artifacts"

    # Model names
    avm_model_name: str = "avm-lgbm"
    recommender_model_name: str = "home-recommender-content"

    # A/B testing
    ab_enabled: bool = True
    ab_traffic_split: float = 0.10  # 10% to challenger

    # Drift thresholds
    psi_warn_threshold: float = 0.10
    psi_alarm_threshold: float = 0.25

    # Feature store
    feature_cache_ttl_seconds: int = 300

    # Serving
    max_recommendations: int = 20

    class Config:
        env_prefix = "REDFIN_ML_"


settings = Settings()


def ensure_dirs() -> None:
    for p in (settings.data_dir, settings.registry_dir, settings.artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)


ensure_dirs()
