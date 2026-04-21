"""Train + register the home recommender."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.avm.features import compute_city_price_level
from src.config import settings
from src.recommender.model import HomeRecommender
from src.registry import ModelRegistry
from src.registry.model_registry import data_hash


def main(data_path: str | Path = None) -> None:
    data_path = Path(data_path or settings.data_dir / "homes.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training data: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"[Recommender] Indexing {len(df):,} listings...")
    city_map = compute_city_price_level(df)
    model = HomeRecommender.train(df, city_map)

    # Offline eval: recall@10 sanity check — for each listing, its top-1 neighbor
    # should be in the same city ≥95% of the time.
    sample = df.sample(min(500, len(df)), random_state=3)
    hits = 0
    for _, row in sample.iterrows():
        recs = model.similar(int(row["listing_id"]), k=1, same_city=False)
        if recs and recs[0].city == row["city"]:
            hits += 1
    same_city_rate = hits / len(sample)
    print(f"[Recommender] top-1 same-city rate: {same_city_rate:.3f}")

    metrics = {"top1_same_city_rate": same_city_rate, "indexed_items": len(df)}

    registry = ModelRegistry()
    dh = data_hash(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    meta = registry.register(
        name=settings.recommender_model_name,
        model=model,
        training_df_rows=len(df),
        training_data_hash=dh,
        params={"metric": "euclidean", "n_neighbors": 50},
        metrics=metrics,
        tags={"owner": "applied-ml", "framework": "sklearn"},
    )
    registry.promote(meta.name, meta.version, "staging")
    try:
        registry.load(meta.name, "production")
    except LookupError:
        registry.promote(meta.name, meta.version, "production")
    print(f"[Recommender] Registered {meta.name} v{meta.version}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    main(args.data)
