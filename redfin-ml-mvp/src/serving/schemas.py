"""Request/response schemas for the serving API.

Separation of concerns: the API contract lives here, not in app.py. Versioning
happens at the URL level (/v1/...) rather than in-body so we can deprecate
cleanly via gateway routing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------- AVM ----------
class HomeFeatures(BaseModel):
    """The feature payload the client sends for a home."""

    city: str = Field(..., examples=["Seattle"])
    property_type: str = Field(..., examples=["single_family"])
    sqft: int = Field(..., gt=0, lt=50_000)
    beds: int = Field(..., ge=0, le=20)
    baths: float = Field(..., ge=0, le=20)
    lot_size: int = Field(..., ge=0, lt=1_000_000)
    year_built: int = Field(..., ge=1800, le=2030)
    garage_spaces: int = Field(0, ge=0, le=10)
    school_score: float = Field(..., ge=1, le=10)
    walk_score: int = Field(..., ge=0, le=100)
    crime_index: float = Field(..., ge=0, le=100)


class PricePrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    point: float
    lower: float
    upper: float
    model_name: str
    model_version: int
    variant: str  # "champion" or "challenger" — from A/B router
    feature_contributions: Dict[str, float]
    request_id: str


class BatchPredictRequest(BaseModel):
    homes: List[HomeFeatures]


class BatchPredictResponse(BaseModel):
    predictions: List[PricePrediction]


# ---------- Recommender ----------
class SimilarHomesRequest(BaseModel):
    listing_id: int
    k: int = Field(10, ge=1, le=50)
    same_city: bool = True


class RecommendedHome(BaseModel):
    listing_id: int
    score: float
    reasons: List[str]
    price: int
    city: str
    sqft: int
    beds: int
    baths: float
    property_type: str


class SimilarHomesResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anchor_listing_id: int
    recommendations: List[RecommendedHome]
    model_name: str
    model_version: int
    request_id: str


# ---------- Agent / AI features ----------
class ExplainRequest(BaseModel):
    features: HomeFeatures
    prediction: PricePrediction


class ExplainResponse(BaseModel):
    narrative: str
    request_id: str


class TriageResponse(BaseModel):
    narrative: str
    overall_severity: str
    max_psi_feature: Optional[str]
    request_id: str


class ParseListingRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000,
                      examples=["Charming 3BR/2BA craftsman in Queen Anne, Seattle. "
                                "1,850 sqft, 2-car garage, built 2003. Walk score 85."])


class ParseListingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    features: HomeFeatures
    prediction: PricePrediction
    narrative: str
    request_id: str


class NLSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       examples=["3BR under $800k near good schools in Seattle"])
    limit: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    listing_id: int
    city: str
    price: int
    beds: int
    baths: float
    sqft: int
    property_type: str
    school_score: float
    year_built: int


class NLSearchResponse(BaseModel):
    query: str
    parsed_filters: dict
    results: List[SearchResult]
    result_count: int
    request_id: str


# ---------- Vector Store ----------
class VectorSearchRequest(BaseModel):
    features: HomeFeatures
    k: int = Field(10, ge=1, le=50)
    city: Optional[str] = None
    max_price: Optional[int] = Field(None, gt=0)
    min_beds: Optional[int] = Field(None, ge=0)
    property_type: Optional[str] = None


class VectorSearchResult(BaseModel):
    listing_id: int
    city: str
    property_type: str
    price: int
    beds: int
    baths: float
    sqft: int
    school_score: float
    walk_score: int
    year_built: int
    similarity_score: float


class VectorSearchResponse(BaseModel):
    query_city: str
    filters_applied: dict
    results: List[VectorSearchResult]
    result_count: int
    index_size: int
    request_id: str


class VectorStatusResponse(BaseModel):
    ready: bool
    index_size: int
    embedding_dims: int
    index_type: str
    persist_dir: str


# ---------- AVM Calibration ----------
class CalibrationBucket(BaseModel):
    label: str
    n: int
    coverage: float     # actual fraction of true prices within CI
    target: float       # nominal CI level (e.g. 0.90)
    well_calibrated: bool


class CalibrationResponse(BaseModel):
    overall_coverage: float
    target_coverage: float
    is_well_calibrated: bool
    mape: float
    median_ape: float
    buckets: List[CalibrationBucket]      # by city / property_type / price_tier
    reliability_points: List[dict]        # [{predicted_conf, actual_conf}]
    n_total: int
    request_id: str


# ---------- What-If Price Sensitivity ----------
class WhatIfRequest(BaseModel):
    features: HomeFeatures
    perturbations: Optional[dict] = Field(
        None,
        description="Map of feature name → new value, e.g. {\"beds\": 4, \"sqft\": 2600}",
        examples=[{"beds": 4, "sqft": 2600}],
    )


class WhatIfResult(BaseModel):
    feature: str
    original_value: float
    new_value: float
    original_price: float
    new_price: float
    delta_dollars: float
    delta_pct: float


class WhatIfResponse(BaseModel):
    base_price: float
    results: List[WhatIfResult]
    request_id: str


# ---------- Fairness Audit ----------
class FairnessSlice(BaseModel):
    group: str
    n: int
    mape: float
    median_ape: float
    mean_price: float
    coverage_90: float


class FairnessResponse(BaseModel):
    overall_mape: float
    slices: List[FairnessSlice]
    max_disparity: float       # max MAPE spread across groups
    disparate_impact_flag: bool
    request_id: str


# ---------- Market Forecast ----------
class ForecastPoint(BaseModel):
    month: int
    label: str             # e.g. "2025-07"
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    city: str
    baseline_price: float
    trend_pct_monthly: float
    price_trend: str       # "rising" | "stable" | "declining"
    points: List[ForecastPoint]
    request_id: str


# ---------- Market Intelligence Agent ----------
class MarketIntelRequest(BaseModel):
    city: str = Field(..., examples=["Seattle"])
    property_type: Optional[str] = Field(None, examples=["single_family"])
    context_notes: Optional[str] = Field(
        None, max_length=500,
        description="Optional context: 'new light rail opening', 'school redistricting', etc.",
        examples=["New Amazon HQ2 campus opening nearby, expected 5k jobs"],
    )


class MarketIntelSignal(BaseModel):
    category: str     # "seasonal" | "development" | "regulatory" | "supply" | "demand"
    signal: str
    impact: str       # "positive" | "negative" | "neutral"
    confidence: str   # "high" | "medium" | "low"


class MarketIntelResponse(BaseModel):
    city: str
    month: str
    narrative: str
    signals: List[MarketIntelSignal]
    price_trend: str     # "rising" | "stable" | "declining"
    best_time_to_buy: str
    best_time_to_sell: str
    request_id: str


# ---------- Monitoring ----------
class DriftReportResponse(BaseModel):
    overall_severity: str
    max_psi: float
    max_psi_feature: Optional[str]
    counts: dict
    features: list


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    avm_version: Optional[int]
    recommender_version: Optional[int]


# ---------- Buyer Preferences (shared by Negotiation + Compare) ----------
class BuyerLifestyle(BaseModel):
    """Lifestyle intent signals that re-weight the match scoring dimensions."""
    household_size: int = Field(1, ge=1, le=12,
                                description="Number of people who will live in the home")
    has_children: bool = Field(False,
                               description="School score weight increases to 55% of neighborhood score")
    has_dogs: bool = Field(False,
                           description="Needs yard (lot_size > 0); condos are penalized")
    work_from_home: bool = Field(False,
                                 description="Extra bedroom beyond min_beds scored as office bonus")
    lifestyle_type: str = Field("suburban",
                                description="'urban' weights walk_score higher; 'suburban' weights lot/safety")
    commute_sensitive: bool = Field(False,
                                    description="Walk score weight increases to 50% of neighborhood score")


class BuyerPreferences(BaseModel):
    max_budget: int = Field(..., gt=0, description="Maximum purchase price in USD")
    min_beds: int = Field(1, ge=0)
    min_baths: float = Field(1.0, ge=0)
    min_school_score: float = Field(5.0, ge=1, le=10)
    min_walk_score: int = Field(0, ge=0, le=100)
    max_crime_index: float = Field(100.0, ge=0, le=100)
    preferred_city: Optional[str] = None
    preferred_property_type: Optional[str] = None
    lifestyle: Optional[BuyerLifestyle] = None


# ---------- Property Comparison ----------
class CompareRequest(BaseModel):
    preferences: BuyerPreferences
    properties: List[HomeFeatures] = Field(..., min_length=2, max_length=5)
    asking_prices: Optional[List[Optional[int]]] = None  # parallel to properties; None = unknown


class DimensionScores(BaseModel):
    budget: float       # 0–100: is AVM within budget?
    size: float         # 0–100: beds/baths match?
    neighborhood: float # 0–100: school + walk + safety
    value: float        # 0–100: AVM vs asking price deal quality
    overall: float      # weighted composite


class PropertyComparison(BaseModel):
    index: int
    city: str
    property_type: str
    avm_estimate: int
    buyer_bid: int
    asking_price: Optional[int]
    scores: DimensionScores
    match_pct: float     # 0–100 overall
    rank: int            # 1 = best match
    pros: List[str]
    cons: List[str]
    winner: bool


class CompareResponse(BaseModel):
    preferences: BuyerPreferences
    properties: List[PropertyComparison]
    winner_index: int
    summary: str


# ---------- Negotiation Intelligence ----------
class NegotiationRequest(BaseModel):
    """Property features + optional market context for bid/offer strategy."""
    city: str = Field(..., examples=["Seattle"])
    property_type: str = Field(..., examples=["single_family"])
    sqft: int = Field(..., gt=0, lt=50_000)
    beds: int = Field(..., ge=0, le=20)
    baths: float = Field(..., ge=0, le=20)
    lot_size: int = Field(0, ge=0)
    year_built: int = Field(..., ge=1800, le=2030)
    garage_spaces: int = Field(0, ge=0, le=10)
    school_score: float = Field(..., ge=1, le=10)
    walk_score: int = Field(..., ge=0, le=100)
    crime_index: float = Field(..., ge=0, le=100)
    # Optional market context — improve strategy if provided
    days_on_market: Optional[int] = Field(None, ge=0, le=1000,
                                          description="How long the listing has been active")
    num_competing_offers: Optional[int] = Field(None, ge=0,
                                                description="Known competing offers (0 = none)")
    asking_price: Optional[int] = Field(None, gt=0,
                                        description="Seller's listed asking price")


class NegotiationFactor(BaseModel):
    factor: str
    impact: str      # "buyer" | "seller" | "neutral"
    detail: str


class NegotiationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    avm_estimate: int
    avm_lower: int
    avm_upper: int
    market_condition: str        # "hot" | "warm" | "cool" | "cold"
    leverage: str                # "buyer" | "neutral" | "seller"
    buyer_bid: int
    buyer_bid_pct_of_avm: float
    seller_list_price: int
    seller_list_pct_of_avm: float
    seller_floor: int
    expected_close_low: int
    expected_close_high: int
    negotiation_gap: int         # seller_list - buyer_bid
    key_factors: List[NegotiationFactor]
    buyer_tactic: str
    seller_tactic: str
    model_version: int
    request_id: str
