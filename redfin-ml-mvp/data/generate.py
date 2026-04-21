"""Synthetic real-estate dataset generator.

We generate ~50k listings with a realistic-ish price process so the AVM
actually has signal to learn. Price is a nonlinear function of sqft, beds,
baths, lot_size, year_built, school_score, walk_score, and a city-level
log-normal price level — plus heteroscedastic noise. Enough structure that
LightGBM learns something useful, but not so much that the model is trivial.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CITIES = {
    # city, base log-price (in log USD), desirability, avg price level
    "Seattle":        (13.55, 1.35),
    "San Francisco":  (13.95, 1.55),
    "Los Angeles":    (13.70, 1.30),
    "Portland":       (13.20, 1.10),
    "Austin":         (13.15, 1.05),
    "Denver":         (13.25, 1.15),
    "Chicago":        (12.90, 0.85),
    "Boston":         (13.50, 1.25),
    "Washington DC":  (13.45, 1.20),
    "Atlanta":        (12.80, 0.80),
    "Dallas":         (12.95, 0.90),
    "Miami":          (13.30, 1.10),
}
PROPERTY_TYPES = ["single_family", "condo", "townhouse", "multi_family"]


def _sample_city_rows(n_total: int) -> pd.Series:
    keys = list(CITIES.keys())
    weights = np.array([CITIES[k][1] for k in keys])
    weights = weights / weights.sum()
    return pd.Series(RNG.choice(keys, size=n_total, p=weights))


def generate(n: int = 50_000, out: str | Path = "data/homes.parquet") -> pd.DataFrame:
    """Generate `n` synthetic homes and write to parquet.

    Returns the DataFrame as well for in-process use.
    """
    city = _sample_city_rows(n)
    ptype = pd.Series(RNG.choice(PROPERTY_TYPES, size=n, p=[0.55, 0.25, 0.15, 0.05]))

    # Core features
    sqft = np.clip(RNG.lognormal(mean=7.45, sigma=0.35, size=n), 350, 9_500)
    beds = np.clip(RNG.integers(1, 6, size=n) + RNG.integers(0, 3, size=n) * (sqft > 3000), 1, 8)
    baths = np.round(np.clip(beds * 0.7 + RNG.normal(0, 0.4, n), 1, 6) * 2) / 2  # half-bath granularity
    lot_size = np.clip(
        RNG.lognormal(mean=8.2, sigma=0.8, size=n) * (ptype == "single_family").astype(int).values
        + RNG.lognormal(mean=6.0, sigma=0.5, size=n) * (ptype != "single_family").astype(int).values,
        300, 50_000,
    )
    year_built = RNG.integers(1900, 2025, size=n)
    garage_spaces = np.clip(RNG.integers(0, 4, size=n), 0, 3)

    # Neighborhood features
    school_score = np.clip(RNG.normal(6.5, 2.0, n), 1, 10)
    walk_score = np.clip(RNG.normal(55, 20, n), 5, 100)
    crime_index = np.clip(RNG.normal(40, 15, n), 5, 95)  # lower = safer

    # Price construction (log-additive)
    base = city.map(lambda c: CITIES[c][0]).to_numpy()
    type_adj = ptype.map({
        "single_family": 0.08,
        "condo": -0.10,
        "townhouse": -0.03,
        "multi_family": 0.12,
    }).to_numpy()

    log_price = (
        base
        + type_adj
        + 0.00018 * sqft
        + 0.03 * beds
        + 0.05 * baths
        + 0.000008 * lot_size
        + 0.0025 * (year_built - 1980)
        + 0.04 * (school_score - 5)
        + 0.006 * (walk_score - 50)
        - 0.008 * (crime_index - 40)
        + 0.02 * garage_spaces
        # Heteroscedastic noise — bigger homes have more price variance
        + RNG.normal(0, 0.10 + 0.00002 * sqft, size=n)
    )
    price = np.round(np.exp(log_price), -2).clip(80_000, 12_000_000)

    df = pd.DataFrame({
        "listing_id": np.arange(1, n + 1),
        "city": city.values,
        "property_type": ptype.values,
        "sqft": np.round(sqft).astype(int),
        "beds": beds.astype(int),
        "baths": baths,
        "lot_size": np.round(lot_size).astype(int),
        "year_built": year_built.astype(int),
        "garage_spaces": garage_spaces.astype(int),
        "school_score": np.round(school_score, 1),
        "walk_score": np.round(walk_score).astype(int),
        "crime_index": np.round(crime_index, 1),
        "price": price.astype(int),
    })

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path.resolve()}")
    print(f"  median price: ${df['price'].median():,.0f}  (range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f})")
    return df


def shift_distribution(df: pd.DataFrame, shift_pct: float = 0.15) -> pd.DataFrame:
    """Return a copy of `df` with feature drift applied (for drift-demo)."""
    out = df.copy()
    out["sqft"] = (out["sqft"] * (1 + shift_pct)).astype(int)
    out["school_score"] = np.clip(out["school_score"] - 1.2, 1, 10)
    out["crime_index"] = np.clip(out["crime_index"] + 8, 5, 95)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--out", default="data/homes.parquet")
    args = parser.parse_args()
    generate(args.n, args.out)
