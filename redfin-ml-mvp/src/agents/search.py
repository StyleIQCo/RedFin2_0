"""Structured home search against the offline feature store.

This is the backend for the natural-language search flow:
  NL query → parse_search_query() → search_homes() → ranked results

In production this would hit a real search index (Elasticsearch, Pinecone,
or a vector store for semantic search). Here we filter the training parquet
directly, which is fast enough at 50k rows for a demo.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.serving.feature_store import feature_store


def search_homes(
    city: Optional[str] = None,
    max_price: Optional[int] = None,
    min_price: Optional[int] = None,
    min_beds: Optional[int] = None,
    max_beds: Optional[int] = None,
    min_baths: Optional[float] = None,
    min_sqft: Optional[int] = None,
    max_sqft: Optional[int] = None,
    property_type: Optional[str] = None,
    min_school_score: Optional[float] = None,
    limit: int = 10,
    **_kwargs,  # absorb unknown keys from LLM output
) -> list[dict]:
    """Filter homes.parquet and return the top `limit` results by price."""
    df: pd.DataFrame = feature_store.get_training_df().copy()

    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if max_price is not None:
        df = df[df["price"] <= max_price]
    if min_price is not None:
        df = df[df["price"] >= min_price]
    if min_beds is not None:
        df = df[df["beds"] >= min_beds]
    if max_beds is not None:
        df = df[df["beds"] <= max_beds]
    if min_baths is not None:
        df = df[df["baths"] >= min_baths]
    if min_sqft is not None:
        df = df[df["sqft"] >= min_sqft]
    if max_sqft is not None:
        df = df[df["sqft"] <= max_sqft]
    if property_type:
        df = df[df["property_type"] == property_type]
    if min_school_score is not None:
        df = df[df["school_score"] >= min_school_score]

    cols = ["listing_id", "city", "price", "beds", "baths", "sqft",
            "property_type", "school_score", "year_built"]
    return (
        df[cols]
        .sort_values("price")
        .head(limit)
        .to_dict("records")
    )
