"""Feature engineering for the AVM.

Key principle: the feature **transform** is the same in training and serving.
That's why this lives in one module and is called by both `train.py` and the
feature store. Training/serving skew is the #1 source of silent AVM bugs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The canonical feature list. Order matters for the LightGBM model.
AVM_FEATURES: list[str] = [
    "sqft",
    "beds",
    "baths",
    "lot_size",
    "age",                  # derived: current_year - year_built
    "garage_spaces",
    "school_score",
    "walk_score",
    "crime_index",
    "city_price_level",     # target-encoded city median price (derived)
    "property_type_code",   # ordinal encoding
    "sqft_per_bed",         # derived interaction
    "is_single_family",
]

PROPERTY_TYPE_CODES = {
    "condo": 0,
    "townhouse": 1,
    "single_family": 2,
    "multi_family": 3,
}


def build_features(df: pd.DataFrame, city_price_level: dict[str, float] | None = None) -> pd.DataFrame:
    """Return a new DataFrame containing AVM_FEATURES only, ready for the model.

    `city_price_level` is the training-time city → median log-price map. We pass
    it from train → serve so we don't recompute at inference and so serving can
    never accidentally use a leaked test-time value.
    """
    out = pd.DataFrame()
    out["sqft"] = df["sqft"].astype(float)
    out["beds"] = df["beds"].astype(float)
    out["baths"] = df["baths"].astype(float)
    out["lot_size"] = df["lot_size"].astype(float)
    out["age"] = (2025 - df["year_built"]).astype(float)
    out["garage_spaces"] = df["garage_spaces"].astype(float)
    out["school_score"] = df["school_score"].astype(float)
    out["walk_score"] = df["walk_score"].astype(float)
    out["crime_index"] = df["crime_index"].astype(float)
    if city_price_level is None:
        # Training path: compute median log-price per city from this df (if it has target).
        # At inference we never hit this — the serving code always passes the map.
        if "price" in df.columns:
            tmp = np.log(df["price"])
            city_price_level = df.assign(_tmp=tmp).groupby("city")["_tmp"].median().to_dict()
        else:
            city_price_level = {}
    out["city_price_level"] = df["city"].map(city_price_level).fillna(
        float(np.mean(list(city_price_level.values())) if city_price_level else 13.0)
    )
    out["property_type_code"] = df["property_type"].map(PROPERTY_TYPE_CODES).fillna(0)
    out["sqft_per_bed"] = out["sqft"] / out["beds"].clip(lower=1)
    out["is_single_family"] = (df["property_type"] == "single_family").astype(float)
    return out[AVM_FEATURES]


def compute_city_price_level(df: pd.DataFrame) -> dict[str, float]:
    """Compute the city → median-log-price map from a labeled training set."""
    return df.assign(_logp=np.log(df["price"])).groupby("city")["_logp"].median().to_dict()
