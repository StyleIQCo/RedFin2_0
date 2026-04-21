"""Fire a burst of shifted requests at a running API to trigger drift.

Usage (API must be running):
    python -m scripts.simulate_drift --n 500 --shift 0.20

Then hit /v1/drift/report to see the alarm trip.
"""
from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from data.generate import shift_distribution
from src.config import settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--shift", type=float, default=0.20)
    args = p.parse_args()

    df = pd.read_parquet(settings.data_dir / "homes.parquet").sample(args.n, random_state=0)
    shifted = shift_distribution(df, shift_pct=args.shift)

    with httpx.Client(timeout=5) as client:
        ok = 0
        t0 = time.perf_counter()
        for _, row in shifted.iterrows():
            payload = {
                "city": row["city"],
                "property_type": row["property_type"],
                "sqft": int(row["sqft"]),
                "beds": int(row["beds"]),
                "baths": float(row["baths"]),
                "lot_size": int(row["lot_size"]),
                "year_built": int(row["year_built"]),
                "garage_spaces": int(row["garage_spaces"]),
                "school_score": float(row["school_score"]),
                "walk_score": int(row["walk_score"]),
                "crime_index": float(row["crime_index"]),
            }
            resp = client.post(f"{args.url}/v1/avm/predict", json=payload)
            if resp.status_code == 200:
                ok += 1
        elapsed = time.perf_counter() - t0
        print(f"Sent {args.n} shifted requests, {ok} OK in {elapsed:.1f}s")

        report = client.get(f"{args.url}/v1/drift/report").json()
        print("\nDrift report:")
        print(f"  severity: {report['overall_severity']}  max PSI: {report['max_psi']:.3f}  feature: {report['max_psi_feature']}")
        print(f"  counts: {report['counts']}")


if __name__ == "__main__":
    main()
