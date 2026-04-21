"""Training/serving skew tests.

The most valuable unit tests on an AVM team: verify that the features computed
in training match the features computed at serving time for the same input.
"""
import numpy as np
import pandas as pd

from src.avm.features import AVM_FEATURES, build_features, compute_city_price_level


def _sample_df():
    return pd.DataFrame([
        {
            "listing_id": 1,
            "city": "Seattle", "property_type": "single_family",
            "sqft": 2000, "beds": 3, "baths": 2.5, "lot_size": 5000,
            "year_built": 1995, "garage_spaces": 2,
            "school_score": 7.5, "walk_score": 80, "crime_index": 30,
            "price": 900_000,
        },
        {
            "listing_id": 2,
            "city": "Austin", "property_type": "condo",
            "sqft": 900, "beds": 1, "baths": 1.0, "lot_size": 400,
            "year_built": 2010, "garage_spaces": 1,
            "school_score": 6.0, "walk_score": 65, "crime_index": 45,
            "price": 450_000,
        },
    ])


def test_build_features_returns_all_expected_columns():
    df = _sample_df()
    city_map = compute_city_price_level(df)
    feats = build_features(df, city_price_level=city_map)
    assert list(feats.columns) == AVM_FEATURES
    assert len(feats) == 2
    assert not feats.isnull().any().any()


def test_features_are_identical_train_vs_serve():
    """Core skew test — same row, same features in both paths."""
    df = _sample_df()
    city_map = compute_city_price_level(df)
    train_feats = build_features(df, city_price_level=city_map)
    # simulate a "single inference" path — one row at a time
    for i in range(len(df)):
        one = df.iloc[[i]]
        serve_feats = build_features(one, city_price_level=city_map)
        np.testing.assert_allclose(
            serve_feats.iloc[0].to_numpy(),
            train_feats.iloc[i].to_numpy(),
            rtol=0, atol=0,
        )


def test_unknown_city_falls_back_to_mean():
    df = _sample_df()
    city_map = compute_city_price_level(df)
    unknown = df.iloc[[0]].copy()
    unknown["city"] = "Atlantis"
    feats = build_features(unknown, city_price_level=city_map)
    mean_val = float(np.mean(list(city_map.values())))
    assert abs(feats.iloc[0]["city_price_level"] - mean_val) < 1e-9
