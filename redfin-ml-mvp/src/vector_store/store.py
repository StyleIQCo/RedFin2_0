"""Chroma-backed vector store for home similarity search.

Why a vector database instead of sklearn's ball tree
-----------------------------------------------------
The existing ANN recommender uses a ball tree over tabular features — fine for
50k rows, but it has two limitations at production scale:

  1. No metadata filtering inside the index.  Filtering by city or price requires
     fetching more candidates than you need and discarding them after the fact.
     Chroma applies the filter *before* the ANN search, so k=10 with a city
     filter costs the same as k=10 without one.

  2. No persistence.  The ball tree lives in RAM and is rebuilt on every restart.
     Chroma writes a persistent HNSW index to disk and hot-loads it in ~200ms
     on startup regardless of dataset size.

At Redfin's scale you'd swap Chroma for Pinecone (managed) or run a Qdrant
cluster for maximum throughput.  The query interface is identical — this module
is the only thing that changes.

Embedding strategy
------------------
We embed each home as a 14-dimensional vector:
  9 normalized continuous features  (sqft, beds, baths, lot, year, garage,
                                     school_score, walk_score, crime_index)
  1 city price level                (captures market tier)
  4 one-hot property type dims

All dimensions are scaled to [0, 1] using min/max stats computed from the
training corpus.  Cosine similarity is used so that scale differences between
features don't dominate — a 1-unit change in beds shouldn't swamp a 100-unit
change in sqft.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

EMBED_FEATURES = [
    "sqft", "beds", "baths", "lot_size", "year_built",
    "garage_spaces", "school_score", "walk_score", "crime_index",
]

PROPERTY_TYPES = ["single_family", "condo", "townhouse", "multi_family"]

CITY_PRICE_LEVEL: Dict[str, float] = {
    "San Francisco": 2.2, "Seattle": 1.6, "Los Angeles": 1.5,
    "Boston": 1.4, "Washington DC": 1.3, "Portland": 1.1,
    "Denver": 1.05, "Austin": 1.0, "Chicago": 0.9,
    "Atlanta": 0.85, "Dallas": 0.85, "Miami": 1.0,
}

_norm_stats: Dict[str, Dict[str, float]] = {}


def _build_embedding(row: dict) -> List[float]:
    """Convert a home feature dict into a normalized 14-dim embedding vector."""
    vec: List[float] = []
    for feat in EMBED_FEATURES:
        val = float(row.get(feat, 0))
        mn = _norm_stats.get(feat, {}).get("min", 0.0)
        mx = _norm_stats.get(feat, {}).get("max", 1.0)
        rng = mx - mn
        vec.append((val - mn) / rng if rng > 0 else 0.0)
    # City price level (normalized to roughly [0,1])
    city_level = CITY_PRICE_LEVEL.get(str(row.get("city", "")), 1.0)
    vec.append(city_level / 2.5)
    # Property type one-hot
    pt = str(row.get("property_type", "single_family"))
    vec.extend([1.0 if pt == t else 0.0 for t in PROPERTY_TYPES])
    return vec


class HomeVectorStore:
    """Chroma-backed persistent vector store with metadata-filtered ANN search."""

    COLLECTION_NAME = "redfin_homes"

    def __init__(self, persist_dir: str = "./redfin_vectors") -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = None
        self._built = False

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self, df) -> int:
        """Index all homes from a DataFrame. Returns number of documents indexed."""
        # Compute normalization stats from corpus
        for feat in EMBED_FEATURES:
            if feat in df.columns:
                _norm_stats[feat] = {
                    "min": float(df[feat].min()),
                    "max": float(df[feat].max()),
                }

        # Recreate collection for a clean build
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        rows = df.to_dict(orient="records")
        batch_size = 500
        total = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            ids        = [str(int(r.get("listing_id", i + j))) for j, r in enumerate(batch)]
            embeddings = [_build_embedding(r) for r in batch]
            metadatas  = [
                {
                    "listing_id":   int(r.get("listing_id", i + j)),
                    "city":         str(r.get("city", "")),
                    "property_type":str(r.get("property_type", "")),
                    "price":        int(r.get("price", 0)),
                    "beds":         int(r.get("beds", 0)),
                    "baths":        float(r.get("baths", 0)),
                    "sqft":         int(r.get("sqft", 0)),
                    "school_score": float(r.get("school_score", 0)),
                    "walk_score":   int(r.get("walk_score", 0)),
                    "year_built":   int(r.get("year_built", 0)),
                }
                for j, r in enumerate(batch)
            ]
            self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            total += len(batch)

        self._built = True
        return total

    def load_existing(self) -> bool:
        """Hot-load a previously persisted index without rebuilding.  Returns True if found."""
        try:
            self._collection = self._client.get_collection(self.COLLECTION_NAME)
            self._built = self._collection.count() > 0
            return self._built
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def similar_by_features(
        self,
        features: dict,
        k: int = 10,
        city: Optional[str] = None,
        max_price: Optional[int] = None,
        min_beds: Optional[int] = None,
        property_type: Optional[str] = None,
    ) -> List[dict]:
        """Find k homes most similar to a feature dict, with optional DB-level filters.

        Filters are evaluated *inside* Chroma's HNSW index — this is the key
        advantage over a post-hoc filter: you always get exactly k results back
        regardless of how selective the filter is.
        """
        if not self._built or self._collection is None:
            return []

        query_vec = _build_embedding(features)
        where = self._build_where(city=city, max_price=max_price,
                                  min_beds=min_beds, property_type=property_type)

        n = min(k, self._collection.count())
        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": n,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return self._format(results, k)

    def similar_by_id(
        self,
        listing_id: int,
        k: int = 10,
        same_city: bool = True,
    ) -> List[dict]:
        """Find homes similar to a listing by ID.  Excludes the query listing itself."""
        if not self._built or self._collection is None:
            return []

        doc = self._collection.get(
            ids=[str(listing_id)],
            include=["embeddings", "metadatas"],
        )
        if not doc["embeddings"]:
            return []

        query_vec  = doc["embeddings"][0]
        city = doc["metadatas"][0].get("city") if same_city and doc["metadatas"] else None
        where = self._build_where(city=city)

        n = min(k + 1, self._collection.count())
        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": n,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        out = [
            r for r in self._format(results, k + 1)
            if int(r.get("listing_id", -1)) != listing_id
        ]
        return out[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        city: Optional[str] = None,
        max_price: Optional[int] = None,
        min_beds: Optional[int] = None,
        property_type: Optional[str] = None,
    ) -> dict:
        conditions = []
        if city:
            conditions.append({"city": {"$eq": city}})
        if max_price:
            conditions.append({"price": {"$lte": max_price}})
        if min_beds:
            conditions.append({"beds": {"$gte": min_beds}})
        if property_type:
            conditions.append({"property_type": {"$eq": property_type}})

        if not conditions:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _format(results: dict, k: int) -> List[dict]:
        out = []
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances",  [[]])[0]
        for meta, dist in zip(metas, distances):
            out.append({**meta, "similarity_score": round(1.0 - float(dist), 4)})
        return out[:k]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._collection.count() if self._collection else 0

    @property
    def is_ready(self) -> bool:
        return self._built and self._collection is not None
