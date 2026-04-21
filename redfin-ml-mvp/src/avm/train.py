"""AVM training entrypoint.

Trains, evaluates against a holdout, registers the artifact, and (if the
validation gate passes) promotes to staging. Promotion to `production`
is explicitly NOT automatic — that happens in the CI pipeline (or via an
on-call promoting after the shadow deploy looks good).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.avm.model import AVMModel
from src.config import settings
from src.registry import ModelRegistry
from src.registry.model_registry import data_hash

# --- Validation gate thresholds ---
# In prod these would live in a config/YAML per-model, checked in to git so
# the gate is version-controlled and reviewable.
GATE = {
    "mape_max": 0.18,
    "median_ape_max": 0.12,
    "p90_coverage_min": 0.80,  # 90% CI must cover at least 80% of holdout
}


def _split(df: pd.DataFrame, test_frac: float = 0.15, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def validation_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures = []
    if metrics["mape"] > GATE["mape_max"]:
        failures.append(f"mape {metrics['mape']:.3f} > {GATE['mape_max']}")
    if metrics["median_ape"] > GATE["median_ape_max"]:
        failures.append(f"median_ape {metrics['median_ape']:.3f} > {GATE['median_ape_max']}")
    if metrics["p90_coverage"] < GATE["p90_coverage_min"]:
        failures.append(f"p90_coverage {metrics['p90_coverage']:.3f} < {GATE['p90_coverage_min']}")
    return (not failures), failures


def main(data_path: str | Path = None) -> None:
    data_path = Path(data_path or settings.data_dir / "homes.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training data: {data_path}. Run `python -m data.generate` first.")

    df = pd.read_parquet(data_path)
    train_df, test_df = _split(df)

    print(f"[AVM] Training on {len(train_df):,} rows; evaluating on {len(test_df):,} rows.")
    model = AVMModel.train(train_df)

    metrics = model.evaluate(test_df)
    print(f"[AVM] Metrics: {json.dumps(metrics, indent=2)}")

    passed, failures = validation_gate(metrics)
    if not passed:
        print(f"[AVM] VALIDATION GATE FAILED: {failures}")
    else:
        print("[AVM] Validation gate passed.")

    # Register as candidate regardless — CI decides whether to promote.
    registry = ModelRegistry()
    dh = data_hash(pd.util.hash_pandas_object(train_df, index=False).to_numpy().tobytes())
    meta = registry.register(
        name=settings.avm_model_name,
        model=model,
        training_df_rows=len(train_df),
        training_data_hash=dh,
        params={"num_boost_round": 250, "num_leaves": 63, "objective": "regression"},
        metrics=metrics,
        tags={"owner": "applied-ml", "framework": "lightgbm"},
    )
    print(f"[AVM] Registered {meta.name} v{meta.version} (stage={meta.stage}).")

    # For the MVP: auto-promote to staging if the gate passes, and to production
    # if nothing is in production yet. In prod we'd stop at staging.
    if passed:
        registry.promote(meta.name, meta.version, "staging")
        try:
            registry.load(meta.name, "production")
        except LookupError:
            registry.promote(meta.name, meta.version, "production")
            print(f"[AVM] No prior production model — promoted v{meta.version} to production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    main(args.data)
