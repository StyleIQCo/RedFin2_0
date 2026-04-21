"""Home recommender — content-based similarity with room for a collab-filter overlay.

The MVP recommender is intentionally simple: an L2-normalized feature matrix
with an approximate-nearest-neighbor index (we use sklearn NearestNeighbors
for simplicity; in prod we'd swap to FAISS/ScaNN).

The design shows two things I'd bring to the Redfin AML team:

  1.  **Hybrid by construction.** The `similar_homes` method uses content
      features. `personalized_for_user` layers a learned user-taste vector
      (stubbed here — would be a matrix-factorization or two-tower model in
      prod) on top of the same index, so we can ship recommendations without
      cold-start issues and improve them as we learn the user.

  2.  **Explainable.** Every recommendation comes with the top features that
      made it similar, so the UI can say "similar because: 3 beds, same city,
      ~1500 sqft, similar school score."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Features used for similarity — a mix of structural + neighborhood signals.
SIM_FEATURES = [
    "sqft", "beds", "baths", "lot_size", "age",
    "school_score", "walk_score", "crime_index",
    "city_price_level", "property_type_code",
]


@dataclass
class Recommendation:
    listing_id: int
    score: float
    reasons: list[str]
    price: int
    city: str
    sqft: int
    beds: int
    baths: float
    property_type: str


def _humanize_reason(feat: str, delta: float) -> str:
    mapping = {
        "sqft": "similar size",
        "beds": "same bedroom count",
        "baths": "similar bathrooms",
        "lot_size": "similar lot",
        "age": "similar age",
        "school_score": "similar school quality",
        "walk_score": "similar walkability",
        "crime_index": "similar neighborhood safety",
        "city_price_level": "similar market tier",
        "property_type_code": "same property type",
    }
    return mapping.get(feat, feat)


class HomeRecommender:
    """ANN over L2-normalized content features; keeps the raw DF for enrichment."""

    def __init__(
        self,
        nn: NearestNeighbors,
        scaler: StandardScaler,
        feature_matrix: np.ndarray,
        listings: pd.DataFrame,
    ) -> None:
        self.nn = nn
        self.scaler = scaler
        self.feature_matrix = feature_matrix
        self.listings = listings.reset_index(drop=True)
        # Fast lookup: listing_id -> row index
        self._id_to_idx = {int(lid): i for i, lid in enumerate(self.listings["listing_id"])}

    @classmethod
    def train(cls, df: pd.DataFrame, city_price_level: dict[str, float]) -> "HomeRecommender":
        from src.avm.features import build_features
        feats = build_features(df, city_price_level=city_price_level)
        feats = feats[SIM_FEATURES].fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(feats.to_numpy())
        # cosine on z-scored features ≈ correlation → good for mixed-scale feats
        nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree", metric="euclidean")
        nn.fit(X)
        keep = df[["listing_id", "city", "property_type", "sqft", "beds", "baths", "price"]].reset_index(drop=True)
        return cls(nn, scaler, X, keep)

    # --- similar homes ---
    def similar(self, listing_id: int, k: int = 10, same_city: bool = True) -> list[Recommendation]:
        if listing_id not in self._id_to_idx:
            raise KeyError(f"Unknown listing_id {listing_id}")
        idx = self._id_to_idx[listing_id]
        dists, neighbors = self.nn.kneighbors(self.feature_matrix[idx: idx + 1], n_neighbors=min(50, len(self.feature_matrix)))
        dists, neighbors = dists[0], neighbors[0]
        anchor_city = self.listings.iloc[idx]["city"]
        out: list[Recommendation] = []
        for dist, n_idx in zip(dists, neighbors):
            if n_idx == idx:
                continue
            if same_city and self.listings.iloc[n_idx]["city"] != anchor_city:
                continue
            row = self.listings.iloc[n_idx]
            score = 1.0 / (1.0 + float(dist))
            reasons = self._explain(idx, n_idx)
            out.append(Recommendation(
                listing_id=int(row["listing_id"]),
                score=score,
                reasons=reasons,
                price=int(row["price"]),
                city=str(row["city"]),
                sqft=int(row["sqft"]),
                beds=int(row["beds"]),
                baths=float(row["baths"]),
                property_type=str(row["property_type"]),
            ))
            if len(out) == k:
                break
        return out

    def _explain(self, anchor_idx: int, neighbor_idx: int, top_n: int = 3) -> list[str]:
        a = self.feature_matrix[anchor_idx]
        b = self.feature_matrix[neighbor_idx]
        diffs = np.abs(a - b)
        # Smallest absolute diffs = most similar features
        nearest_feats = np.argsort(diffs)[:top_n]
        return [_humanize_reason(SIM_FEATURES[i], float(diffs[i])) for i in nearest_feats]

    def personalized(self, user_vector: np.ndarray, k: int = 10, city_filter: str | None = None) -> list[Recommendation]:
        """Cold-start-safe personalization stub.

        `user_vector` has the same shape as a single feature row — in prod it
        would be learned from user activity (favorites/views) via two-tower
        embeddings. Here we just treat it as a synthetic "ideal home" query.
        """
        if user_vector.shape != (len(SIM_FEATURES),):
            raise ValueError(f"user_vector must be shape ({len(SIM_FEATURES)},)")
        scaled = self.scaler.transform(user_vector.reshape(1, -1))
        dists, neighbors = self.nn.kneighbors(scaled, n_neighbors=min(50, len(self.feature_matrix)))
        dists, neighbors = dists[0], neighbors[0]
        out: list[Recommendation] = []
        for dist, n_idx in zip(dists, neighbors):
            row = self.listings.iloc[n_idx]
            if city_filter and row["city"] != city_filter:
                continue
            out.append(Recommendation(
                listing_id=int(row["listing_id"]),
                score=1.0 / (1.0 + float(dist)),
                reasons=["matches your profile"],
                price=int(row["price"]),
                city=str(row["city"]),
                sqft=int(row["sqft"]),
                beds=int(row["beds"]),
                baths=float(row["baths"]),
                property_type=str(row["property_type"]),
            ))
            if len(out) == k:
                break
        return out
