"""In-process feature store stub.

A real feature store (Feast, Tecton, SageMaker Feature Store) is its own
platform. The job of the stub here is to make one thing obvious:

    **Training reads and serving reads come through the same interface.**

That's how you prevent training/serving skew. In training, the feature store
serves from an offline/batch store (Parquet on S3, or the data warehouse). In
serving, it serves from an online/low-latency store (Redis/DynamoDB). The
transforms in between are identical — they live in `src/avm/features.py`.

Here the "online" store is just a dict, and the "offline" store is the
training parquet. The surface area is what matters for the interview conversation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings


@dataclass
class FeatureRow:
    listing_id: int
    features: dict[str, Any]
    fetched_from: str  # "cache" | "online_store" | "offline_store"
    stale_seconds: float


class FeatureStore:
    def __init__(self, offline_parquet: Path | None = None) -> None:
        self._offline_path = offline_parquet or (settings.data_dir / "homes.parquet")
        self._offline_df: pd.DataFrame | None = None
        # "Online" store — a simple dict; TTL cached.
        self._online: dict[int, tuple[dict, float]] = {}
        self._cache_ttl = settings.feature_cache_ttl_seconds

    # --- offline (batch / training) ---
    def get_training_df(self) -> pd.DataFrame:
        """Load the offline feature table — the same data the AVM trained on."""
        if self._offline_df is None:
            if not self._offline_path.exists():
                raise FileNotFoundError(f"Offline store missing: {self._offline_path}")
            self._offline_df = pd.read_parquet(self._offline_path)
        return self._offline_df

    def seed_online(self) -> int:
        """Hydrate the online store from the offline table.

        In production this is a batch materialization job that writes to
        Redis/DynamoDB on a schedule (hourly for hot features, nightly for
        cold). Here we do it at startup so the API has data to serve.
        """
        df = self.get_training_df()
        now = time.time()
        for row in df.itertuples(index=False):
            self._online[int(row.listing_id)] = (row._asdict(), now)
        return len(self._online)

    # --- online (serving) ---
    def get_online(self, listing_id: int) -> FeatureRow:
        rec = self._online.get(listing_id)
        if rec is None:
            raise KeyError(f"No features for listing_id {listing_id}")
        features, fetched_at = rec
        age = time.time() - fetched_at
        return FeatureRow(
            listing_id=listing_id,
            features=features,
            fetched_from="online_store",
            stale_seconds=age,
        )

    def get_online_many(self, listing_ids: list[int]) -> list[FeatureRow]:
        return [self.get_online(i) for i in listing_ids if i in self._online]


# Module-level singleton — the API imports this once.
feature_store = FeatureStore()
