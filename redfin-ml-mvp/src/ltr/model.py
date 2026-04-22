"""Learning-to-Rank using LightGBM LambdaRank.

Ranks home listings by relevance to a buyer's search query.
Training labels are derived from buyer-preference match scores (relevance 0–3).
At inference, candidate listings are scored and sorted by predicted relevance.

This is the core ML model powering Redfin's search results ordering.
Filtering (price < $800k) is necessary but not sufficient — LTR determines
which of the 200 matching homes appears at the top.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

QUERY_FEATURES = ["budget_m", "min_beds", "min_baths", "min_school_score", "min_walk_score"]

DOC_FEATURES = [
    "price_m", "beds", "baths", "sqft_k", "year_norm",
    "school_score", "walk_norm", "crime_norm", "pt_encoded",
]

INTERACTION_FEATURES = [
    "budget_ratio", "bed_excess", "bath_excess", "school_excess", "affordability",
]

ALL_FEATURES = QUERY_FEATURES + DOC_FEATURES + INTERACTION_FEATURES

PT_ENCODE = {"single_family": 0, "condo": 1, "townhouse": 2, "multi_family": 3}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_features(query: dict, listing: pd.Series) -> list:
    """Return a 19-element feature vector for (query, listing) pair."""
    budget = query["budget"]
    min_beds = query["min_beds"]
    min_baths = query["min_baths"]
    min_school = query["min_school_score"]
    min_walk = query["min_walk_score"]

    price = listing["price"]
    beds = listing["beds"]
    baths = listing["baths"]
    sqft = listing["sqft"]
    year = listing["year_built"]
    school = listing["school_score"]
    walk = listing["walk_score"]
    crime = listing["crime_index"]
    ptype = listing.get("property_type", "single_family")

    # Query features
    budget_m = budget / 1e6
    # Document features
    price_m = price / 1e6
    sqft_k = sqft / 1000
    year_norm = year / 2024
    walk_norm = walk / 100
    crime_norm = crime / 100
    pt_encoded = PT_ENCODE.get(ptype, 0)

    # Interaction features
    budget_ratio = price / budget if budget > 0 else 1.0
    bed_excess = beds - min_beds
    bath_excess = baths - min_baths
    school_excess = school - min_school
    affordability = budget / price if price > 0 else 1.0

    return [
        # Query
        budget_m, min_beds, min_baths, min_school, min_walk,
        # Document
        price_m, beds, baths, sqft_k, year_norm,
        school, walk_norm, crime_norm, pt_encoded,
        # Interactions
        budget_ratio, bed_excess, bath_excess, school_excess, affordability,
    ]


# ---------------------------------------------------------------------------
# Relevance labelling
# ---------------------------------------------------------------------------

def compute_relevance(query: dict, listing: pd.Series) -> int:
    """Return an integer relevance label in [0, 3] for a (query, listing) pair."""
    budget = query["budget"]
    price = listing["price"]

    score = 0

    # Price signal (up to 2 points + 1 bonus)
    if price <= budget:
        score += 2
        if price <= budget * 0.85:
            score += 1  # bonus for well within budget
    elif price <= budget * 1.05:
        score += 1  # partial credit: just over budget

    # Attribute matches
    if listing["beds"] >= query["min_beds"]:
        score += 1
    if listing["baths"] >= query["min_baths"]:
        score += 1
    if listing["school_score"] >= query["min_school_score"]:
        score += 1
    if listing["walk_score"] >= query["min_walk_score"]:
        score += 1

    return min(score, 3)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k for a single query group."""
    order = np.argsort(scores)[::-1][:k]
    ranked_labels = labels[order]
    ideal_labels = np.sort(labels)[::-1][:k]

    def dcg(rels: np.ndarray) -> float:
        gains = (2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))
        return gains.sum()

    idcg = dcg(ideal_labels)
    return dcg(ranked_labels) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------

def _top_factors(query: dict, listing: pd.Series) -> list[str]:
    """Return 2–3 human-readable strings explaining the listing's rank position."""
    factors = []
    price = listing["price"]
    budget = query["budget"]

    gap = price - budget
    if gap > 0:
        factors.append(f"${gap:,.0f} over budget")
    elif gap < 0:
        factors.append(f"${abs(gap):,.0f} under budget")
    else:
        factors.append("exactly at budget")

    bed_diff = listing["beds"] - query["min_beds"]
    if bed_diff >= 0:
        factors.append(f"beds: {listing['beds']} (need {query['min_beds']})")
    else:
        factors.append(f"beds short by {abs(bed_diff)}")

    school_diff = listing["school_score"] - query["min_school_score"]
    if school_diff >= 0:
        factors.append(f"school {listing['school_score']:.1f} ≥ min {query['min_school_score']:.1f}")
    else:
        factors.append(f"school {listing['school_score']:.1f} below min {query['min_school_score']:.1f}")

    return factors[:3]


# ---------------------------------------------------------------------------
# LTRRanker
# ---------------------------------------------------------------------------

class LTRRanker:
    """LightGBM LambdaRank model for ranking real-estate listings."""

    model: Optional[lgb.LGBMRanker] = None
    ndcg_at_10: float = 0.0
    feature_importances: dict = {}

    def __init__(self) -> None:
        self.model = None
        self.ndcg_at_10 = 0.0
        self.feature_importances = {}

    @classmethod
    def train(cls, df: pd.DataFrame, n_queries_per_city: int = 15) -> "LTRRanker":
        """Train a LambdaRank model on synthetic buyer queries over `df`."""
        rng = np.random.default_rng(42)

        all_features: list[list] = []
        all_labels: list[int] = []
        all_groups: list[int] = []  # number of docs per query

        for city in df["city"].unique():
            city_df = df[df["city"] == city].reset_index(drop=True)
            sample_size = min(300, len(city_df))
            city_sample = city_df.sample(n=sample_size, random_state=42)
            median_price = city_sample["price"].median()

            budget_multipliers = [0.7, 0.9, 1.1, 1.3]
            min_beds_choices = [1, 2, 3, 4]
            min_baths_choices = [1.0, 1.5, 2.0, 2.5]
            min_school_choices = [5.0, 6.5, 7.5, 8.5]
            min_walk_choices = [0, 30, 60, 80]

            for _ in range(n_queries_per_city):
                mult = rng.choice(budget_multipliers)
                query = {
                    "budget": float(median_price * mult),
                    "min_beds": int(rng.choice(min_beds_choices)),
                    "min_baths": float(rng.choice(min_baths_choices)),
                    "min_school_score": float(rng.choice(min_school_choices)),
                    "min_walk_score": int(rng.choice(min_walk_choices)),
                }

                feats = [make_features(query, row) for _, row in city_sample.iterrows()]
                labels = [compute_relevance(query, row) for _, row in city_sample.iterrows()]

                all_features.extend(feats)
                all_labels.extend(labels)
                all_groups.append(len(feats))

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        groups = np.array(all_groups, dtype=np.int32)

        # 80/20 split at the query (group) boundary
        n_train_groups = int(len(groups) * 0.8)
        train_size = int(groups[:n_train_groups].sum())

        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        g_train, g_val = groups[:n_train_groups], groups[n_train_groups:]

        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            lambdarank_truncation_level=10,
            verbose=-1,
        )
        ranker.fit(X_train, y_train, group=g_train)

        # Evaluate NDCG@10 on holdout
        val_preds = ranker.predict(X_val)
        ndcg_scores = []
        offset = 0
        for g_size in g_val:
            grp_labels = y_val[offset: offset + g_size]
            grp_scores = val_preds[offset: offset + g_size]
            ndcg_scores.append(_ndcg_at_k(grp_labels, grp_scores, k=10))
            offset += g_size

        instance = cls()
        instance.model = ranker
        instance.ndcg_at_10 = float(np.mean(ndcg_scores))
        instance.feature_importances = dict(
            sorted(
                zip(ALL_FEATURES, ranker.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        return instance

    def rank(self, query: dict, candidates: list[dict]) -> list[dict]:
        """Score and rank `candidates` for `query`. Returns enriched dicts."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call LTRRanker.train() first.")

        rows = [pd.Series(c) for c in candidates]
        X = np.array([make_features(query, r) for r in rows], dtype=np.float32)
        scores = self.model.predict(X)

        enriched = []
        for i, (candidate, score, row) in enumerate(
            sorted(zip(candidates, scores, rows), key=lambda t: t[1], reverse=True)
        ):
            item = dict(candidate)
            item["relevance_score"] = float(score)
            item["rank"] = i + 1
            item["rank_factors"] = _top_factors(query, row)
            enriched.append(item)

        return enriched

    def save(self, path: Path) -> None:
        """Persist the ranker to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LTRRanker":
        """Load a persisted ranker from disk."""
        return joblib.load(path)
