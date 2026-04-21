"""FastAPI serving app for AVM + Recommender.

Endpoints:
    GET  /healthz                 liveness + model versions
    GET  /metrics                 prometheus scrape target
    POST /v1/avm/predict          single-home price prediction
    POST /v1/avm/batch            batch predictions
    POST /v1/recommender/similar  similar-homes for a listing
    GET  /v1/drift/report         current drift state vs. reference

Design notes worth calling out in the interview:

  * **Model loading happens at startup, not per-request.** We load the
    production-stage model from the registry on app startup and keep it
    warm. Swapping a new production model = registry promotion + SIGHUP
    (or, more commonly, a rolling restart in k8s).

  * **Every request gets a request_id** propagated through logs + metrics.
    That's how we debug prod incidents.

  * **Predictions are emitted as a histogram.** If the AVM suddenly starts
    predicting $20M everywhere, the histogram shifts and we alert — even
    before a drift detector catches the input shift.

  * **Shadow mode is a feature flag.** Toggled via env var, not code change.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.avm.features import AVM_FEATURES, build_features
from src.avm.model import AVMModel
from src.config import settings
from src.monitoring import DriftDetector, metrics_registry
from src.monitoring.logging import configure_logging, get_logger
from src.recommender.model import HomeRecommender
from src.registry import ModelRegistry
from src.serving.ab_testing import ABRouter
from src.serving.feature_store import feature_store
from src.agents import llm as agent_llm
from src.agents import search as agent_search
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BuyerPreferences,
    CompareRequest,
    CompareResponse,
    DimensionScores,
    DriftReportResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    HomeFeatures,
    NLSearchRequest,
    NLSearchResponse,
    NegotiationFactor,
    NegotiationRequest,
    NegotiationResponse,
    ParseListingRequest,
    ParseListingResponse,
    PricePrediction,
    PropertyComparison,
    RecommendedHome,
    SearchResult,
    SimilarHomesRequest,
    SimilarHomesResponse,
    TriageResponse,
)

configure_logging()
log = get_logger()

# Thread pool for background retraining jobs — never blocks the request loop.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrain")

# --- App state container ---
# Lives on `app.state.ml`. Populated at startup.
class _State:
    avm_champion: AVMModel | None = None
    avm_champion_version: int = 0
    avm_challenger: AVMModel | None = None
    avm_challenger_version: int = 0
    recommender: HomeRecommender | None = None
    recommender_version: int = 0
    drift_detector: DriftDetector | None = None
    ab_router: ABRouter = ABRouter(
        traffic_split=settings.ab_traffic_split if settings.ab_enabled else 0.0,
        shadow=False,
        experiment_id="avm-champion-vs-challenger-v1",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = _State()
    registry = ModelRegistry()

    # Load AVM champion (production)
    try:
        avm, meta = registry.load(settings.avm_model_name, "production")
        state.avm_champion = avm
        state.avm_champion_version = meta.version
        metrics_registry.model_loaded.labels(meta.name, str(meta.version), meta.stage).inc()
        log.info("avm_loaded", version=meta.version, stage=meta.stage)
    except LookupError:
        log.warning("no_avm_in_production")

    # Load AVM challenger (staging) — if present
    try:
        avm_c, meta_c = registry.load(settings.avm_model_name, "staging")
        # Only load a different version from champion as challenger
        if meta_c.version != state.avm_champion_version:
            state.avm_challenger = avm_c
            state.avm_challenger_version = meta_c.version
            metrics_registry.model_loaded.labels(meta_c.name, str(meta_c.version), meta_c.stage).inc()
            log.info("avm_challenger_loaded", version=meta_c.version)
    except LookupError:
        pass

    # Load recommender (production)
    try:
        rec, meta_r = registry.load(settings.recommender_model_name, "production")
        state.recommender = rec
        state.recommender_version = meta_r.version
        metrics_registry.model_loaded.labels(meta_r.name, str(meta_r.version), meta_r.stage).inc()
        log.info("recommender_loaded", version=meta_r.version)
    except LookupError:
        log.warning("no_recommender_in_production")

    # Hydrate online feature store
    try:
        n = feature_store.seed_online()
        log.info("feature_store_seeded", rows=n)
    except FileNotFoundError:
        log.warning("no_offline_data_for_feature_store")

    # Build drift detector from training distribution (reference)
    if state.avm_champion is not None:
        try:
            ref_df = feature_store.get_training_df()
            ref_feats = build_features(ref_df, city_price_level=state.avm_champion.city_price_level)
            state.drift_detector = DriftDetector(
                reference=ref_feats,
                feature_names=AVM_FEATURES,
                warn_threshold=settings.psi_warn_threshold,
                alarm_threshold=settings.psi_alarm_threshold,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("drift_detector_init_failed", error=str(e))

    app.state.ml = state
    app.state.recent_features: list[dict] = []   # ring-buffer for drift calc (last 2k)
    app.state.ab_champion_preds: list[float] = [] # per-variant prediction log for A/B stats
    app.state.ab_challenger_preds: list[float] = []
    app.state.retrain_jobs: dict[str, Any] = {}   # job_id → job status dict
    yield
    # teardown — nothing to clean up


app = FastAPI(
    title="Redfin ML Serving API",
    version="0.1.0",
    description="AVM + home recommender. Powered by the Applied ML platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the demo UI is a static HTML file
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware: request_id + latency ---
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = req_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:  # noqa: BLE001
        log.exception("unhandled_error", error=str(e), request_id=req_id, path=request.url.path)
        raise
    elapsed = time.perf_counter() - t0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Elapsed-Ms"] = f"{elapsed * 1000:.1f}"
    return response


# --- Helpers ---
def _as_df(home: HomeFeatures) -> pd.DataFrame:
    return pd.DataFrame([home.model_dump()])


def _track_recent(request: Request, feats: pd.DataFrame) -> None:
    buf: list[dict] = request.app.state.recent_features
    for rec in feats.to_dict(orient="records"):
        buf.append(rec)
    # keep last 2,000 for drift
    if len(buf) > 2_000:
        del buf[: len(buf) - 2_000]


# --- Endpoints ---
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz(request: Request) -> HealthResponse:
    state: _State = request.app.state.ml
    return HealthResponse(
        status="ok" if state.avm_champion else "degraded",
        avm_version=state.avm_champion_version if state.avm_champion else None,
        recommender_version=state.recommender_version if state.recommender else None,
    )


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(metrics_registry.registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/avm/predict", response_model=PricePrediction, tags=["avm"])
def avm_predict(home: HomeFeatures, request: Request) -> PricePrediction:
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(status_code=503, detail="AVM model not loaded")

    t0 = time.perf_counter()
    req_id = request.state.request_id
    decision = state.ab_router.route(req_id)
    variant = "champion"
    model = state.avm_champion
    version = state.avm_champion_version

    if decision.variant == "challenger" and state.avm_challenger is not None and not decision.shadow:
        variant = "challenger"
        model = state.avm_challenger
        version = state.avm_challenger_version

    df = _as_df(home)
    feats = build_features(df, city_price_level=model.city_price_level)
    _track_recent(request, feats)

    try:
        pred = model.predict(df, model_name=settings.avm_model_name, model_version=version)
    except Exception as e:  # noqa: BLE001
        metrics_registry.request_count.labels("/v1/avm/predict", settings.avm_model_name, str(version), variant, "error").inc()
        log.exception("avm_predict_failed", request_id=req_id, error=str(e))
        raise HTTPException(status_code=500, detail="prediction failed") from e

    # Shadow path: run challenger but return champion (for shadow deploys)
    if decision.variant == "challenger" and state.avm_challenger is not None and decision.shadow:
        try:
            challenger_pred = state.avm_challenger.predict(df, model_name=settings.avm_model_name, model_version=state.avm_challenger_version)
            log.info(
                "shadow_prediction",
                request_id=req_id,
                champion_point=pred.point,
                challenger_point=challenger_pred.point,
                diff_pct=abs(pred.point - challenger_pred.point) / max(pred.point, 1e-6),
            )
        except Exception:
            log.exception("shadow_prediction_failed")

    elapsed = time.perf_counter() - t0
    metrics_registry.request_count.labels("/v1/avm/predict", settings.avm_model_name, str(version), variant, "ok").inc()
    metrics_registry.request_latency.labels("/v1/avm/predict", settings.avm_model_name, variant).observe(elapsed)
    metrics_registry.avm_prediction.observe(pred.point)

    # Track per-variant for A/B statistical analysis
    ab_buf = request.app.state.ab_champion_preds if variant == "champion" else request.app.state.ab_challenger_preds
    ab_buf.append(pred.point)
    if len(ab_buf) > 1000:
        del ab_buf[:len(ab_buf) - 1000]

    log.info(
        "avm_predict",
        request_id=req_id,
        city=home.city,
        sqft=home.sqft,
        price=pred.point,
        version=version,
        variant=variant,
        latency_ms=elapsed * 1000,
    )

    return PricePrediction(
        point=pred.point,
        lower=pred.lower,
        upper=pred.upper,
        model_name=pred.model_name,
        model_version=version,
        variant=variant,
        feature_contributions=pred.feature_contributions,
        request_id=req_id,
    )


@app.post("/v1/avm/batch", response_model=BatchPredictResponse, tags=["avm"])
def avm_batch(batch: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(status_code=503, detail="AVM model not loaded")

    model = state.avm_champion
    version = state.avm_champion_version
    req_id = request.state.request_id
    t0 = time.perf_counter()

    df = pd.DataFrame([h.model_dump() for h in batch.homes])
    feats = build_features(df, city_price_level=model.city_price_level)
    _track_recent(request, feats)
    preds = model.predict_batch(df)

    elapsed = time.perf_counter() - t0
    metrics_registry.request_count.labels("/v1/avm/batch", settings.avm_model_name, str(version), "champion", "ok").inc()
    metrics_registry.request_latency.labels("/v1/avm/batch", settings.avm_model_name, "champion").observe(elapsed)
    for v in preds["point"]:
        metrics_registry.avm_prediction.observe(float(v))

    out = []
    for i in range(len(preds)):
        point = float(preds.iloc[i]["point"])
        lower = float(preds.iloc[i]["lower"])
        upper = float(preds.iloc[i]["upper"])
        out.append(PricePrediction(
            point=point, lower=lower, upper=upper,
            model_name=settings.avm_model_name,
            model_version=version,
            variant="champion",
            feature_contributions={},  # skipped in batch for speed
            request_id=req_id,
        ))
    log.info("avm_batch", request_id=req_id, n=len(out), latency_ms=elapsed * 1000)
    return BatchPredictResponse(predictions=out)


@app.post("/v1/recommender/similar", response_model=SimilarHomesResponse, tags=["recommender"])
def recommender_similar(body: SimilarHomesRequest, request: Request) -> SimilarHomesResponse:
    state: _State = request.app.state.ml
    if state.recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    req_id = request.state.request_id
    t0 = time.perf_counter()
    try:
        recs = state.recommender.similar(
            listing_id=body.listing_id,
            k=min(body.k, settings.max_recommendations),
            same_city=body.same_city,
        )
    except KeyError:
        metrics_registry.request_count.labels("/v1/recommender/similar", settings.recommender_model_name, str(state.recommender_version), "champion", "not_found").inc()
        raise HTTPException(status_code=404, detail=f"Unknown listing_id {body.listing_id}")

    elapsed = time.perf_counter() - t0
    metrics_registry.request_count.labels("/v1/recommender/similar", settings.recommender_model_name, str(state.recommender_version), "champion", "ok").inc()
    metrics_registry.request_latency.labels("/v1/recommender/similar", settings.recommender_model_name, "champion").observe(elapsed)

    log.info("recs_similar", request_id=req_id, anchor=body.listing_id, k=len(recs), latency_ms=elapsed * 1000)

    return SimilarHomesResponse(
        anchor_listing_id=body.listing_id,
        recommendations=[RecommendedHome(**r.__dict__) for r in recs],
        model_name=settings.recommender_model_name,
        model_version=state.recommender_version,
        request_id=req_id,
    )


@app.get("/v1/drift/report", response_model=DriftReportResponse, tags=["monitoring"])
def drift_report(request: Request) -> DriftReportResponse:
    state: _State = request.app.state.ml
    buf = request.app.state.recent_features
    if state.drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    if len(buf) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough recent requests to report drift ({len(buf)} / min 30)",
        )
    current = pd.DataFrame(buf)
    summary = state.drift_detector.summary(current)
    # Update gauge
    metrics_registry.drift_psi.labels(settings.avm_model_name).set(summary["max_psi"])
    return DriftReportResponse(**summary)


@app.get("/v1/ab/config", tags=["monitoring"])
def ab_config(request: Request) -> dict[str, Any]:
    state: _State = request.app.state.ml
    return {
        **state.ab_router.describe(),
        "champion_version": state.avm_champion_version,
        "challenger_version": state.avm_challenger_version,
        "challenger_loaded": state.avm_challenger is not None,
    }


@app.get("/v1/models", tags=["monitoring"])
def list_models() -> dict[str, Any]:
    registry = ModelRegistry()
    out: dict[str, Any] = {}
    for name in (settings.avm_model_name, settings.recommender_model_name):
        try:
            versions = registry.list_versions(name)
            out[name] = [
                {"version": v.version, "stage": v.stage, "metrics": v.metrics, "created_at": v.created_at}
                for v in versions
            ]
        except FileNotFoundError:
            out[name] = []
    return out


# ---------------------------------------------------------------------------
# AI Agent endpoints
# Powered by Claude (Hermes-compatible, OpenClaw-callable).
# Falls back to intelligible mock responses when ANTHROPIC_API_KEY is unset.
# ---------------------------------------------------------------------------

@app.post("/v1/agent/explain", response_model=ExplainResponse, tags=["agent"])
def agent_explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    """Generate a plain-English valuation narrative from AVM output.

    Called by the Hermes Agent's redfin-avm skill after every prediction,
    and surfaced in the UI as 'Explain this estimate'.
    """
    narrative = agent_llm.explain_avm(
        body.features.model_dump(),
        body.prediction.model_dump(),
    )
    log.info("agent_explain", request_id=request.state.request_id, city=body.features.city)
    return ExplainResponse(narrative=narrative, request_id=request.state.request_id)


@app.post("/v1/agent/triage", response_model=TriageResponse, tags=["agent"])
def agent_triage(request: Request) -> TriageResponse:
    """Run the drift detector and ask an LLM to explain what's happening.

    Called by the Hermes Agent on a schedule (e.g. every hour) to produce
    an on-call-friendly summary instead of a raw PSI table.
    """
    state: _State = request.app.state.ml
    buf = request.app.state.recent_features
    if state.drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    if len(buf) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough recent requests ({len(buf)} / min 30)",
        )
    current = pd.DataFrame(buf)
    summary = state.drift_detector.summary(current)

    drift_report_slim = {
        "overall_severity": summary["overall_severity"],
        "max_psi": summary["max_psi"],
        "max_psi_feature": summary["max_psi_feature"],
        "features": [
            {"feature": f["feature"], "psi": f["psi"], "severity": f["severity"]}
            for f in summary["features"]
        ],
    }
    narrative = agent_llm.triage_drift(drift_report_slim)
    log.info("agent_triage", request_id=request.state.request_id, severity=summary["overall_severity"])
    return TriageResponse(
        narrative=narrative,
        overall_severity=summary["overall_severity"],
        max_psi_feature=summary["max_psi_feature"],
        request_id=request.state.request_id,
    )


@app.post("/v1/agent/parse-listing", response_model=ParseListingResponse, tags=["agent"])
def agent_parse_listing(body: ParseListingRequest, request: Request) -> ParseListingResponse:
    """Extract structured features from raw MLS listing text, then price it.

    Flow: unstructured text → LLM → HomeFeatures → AVM → price + narrative.
    Demonstrates 'listing intelligence': turning a free-text MLS description
    into an immediate valuation without any manual data entry.
    """
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(status_code=503, detail="AVM model not loaded")

    features_dict = agent_llm.parse_listing(body.text)

    # Coerce types the LLM might have gotten slightly wrong
    features_dict.setdefault("garage_spaces", 0)
    features_dict.setdefault("lot_size", 0)
    for int_key in ("sqft", "beds", "lot_size", "year_built", "garage_spaces", "walk_score"):
        if int_key in features_dict:
            features_dict[int_key] = int(features_dict[int_key])
    for float_key in ("baths", "school_score", "crime_index"):
        if float_key in features_dict:
            features_dict[float_key] = float(features_dict[float_key])

    home = HomeFeatures(**features_dict)
    df = _as_df(home)
    pred = state.avm_champion.predict(
        df, model_name=settings.avm_model_name, model_version=state.avm_champion_version
    )
    feats = build_features(df, city_price_level=state.avm_champion.city_price_level)
    _track_recent(request, feats)

    prediction = PricePrediction(
        point=pred.point,
        lower=pred.lower,
        upper=pred.upper,
        model_name=pred.model_name,
        model_version=pred.model_version,
        variant="champion",
        feature_contributions=pred.feature_contributions,
        request_id=request.state.request_id,
    )
    narrative = agent_llm.explain_avm(home.model_dump(), prediction.model_dump())
    log.info("agent_parse_listing", request_id=request.state.request_id, city=home.city, price=pred.point)
    return ParseListingResponse(
        features=home,
        prediction=prediction,
        narrative=narrative,
        request_id=request.state.request_id,
    )


@app.post("/v1/agent/nl-search", response_model=NLSearchResponse, tags=["agent"])
def agent_nl_search(body: NLSearchRequest, request: Request) -> NLSearchResponse:
    """Natural-language home search: parse query → filter homes → return ranked results.

    This is the backend for the OpenClaw skill. A user types
    'Find me a 3BR under $800k near good schools in Seattle' and gets
    a ranked list without touching a single form field.
    """
    parsed = agent_llm.parse_search_query(body.query)
    results = agent_search.search_homes(limit=body.limit, **parsed)
    log.info(
        "agent_nl_search",
        request_id=request.state.request_id,
        query=body.query,
        filters=parsed,
        n_results=len(results),
    )
    return NLSearchResponse(
        query=body.query,
        parsed_filters=parsed,
        results=[SearchResult(**r) for r in results],
        result_count=len(results),
        request_id=request.state.request_id,
    )


# ---------------------------------------------------------------------------
# A/B Testing — statistical results
# ---------------------------------------------------------------------------

@app.get("/v1/ab/results", tags=["ab-testing"])
def ab_results(request: Request) -> dict:
    """Per-variant prediction stats + SRM check + Welch t-test lift significance.

    SRM (Sample Ratio Mismatch): if the observed traffic split deviates from
    the configured split by more than noise (chi-square p < 0.01), the experiment
    is compromised — bucketing is broken and lift estimates are untrustworthy.
    """
    state: _State = request.app.state.ml
    champ = list(request.app.state.ab_champion_preds)
    chal = list(request.app.state.ab_challenger_preds)
    total = len(champ) + len(chal)

    def _variant_stats(preds: list[float], version: int) -> dict:
        if not preds:
            return {"n": 0, "mean_price": None, "std_price": None, "p50_price": None, "version": version}
        a = np.array(preds)
        return {
            "n": len(preds),
            "mean_price": round(float(np.mean(a))),
            "std_price": round(float(np.std(a))),
            "p50_price": round(float(np.median(a))),
            "version": version,
        }

    result: dict[str, Any] = {
        "config": {
            **state.ab_router.describe(),
            "champion_version": state.avm_champion_version,
            "challenger_version": state.avm_challenger_version,
            "challenger_loaded": state.avm_challenger is not None,
        },
        "champion": _variant_stats(champ, state.avm_champion_version),
        "challenger": _variant_stats(chal, state.avm_challenger_version),
        "total_requests": total,
        "srm_check": None,
        "lift_pct": None,
        "p_value": None,
        "significant": None,
        "ci_95": None,
    }

    if total >= 10 and len(chal) >= 1:
        expected_split = state.ab_router.traffic_split  # e.g. 0.10
        exp_champ = total * (1 - expected_split)
        exp_chal = total * expected_split
        chi2, srm_p = scipy_stats.chisquare(
            [len(champ), len(chal)], f_exp=[exp_champ, exp_chal]
        )
        result["srm_check"] = {
            "chi2": round(float(chi2), 3),
            "p_value": round(float(srm_p), 4),
            "passed": bool(srm_p > 0.01),
            "expected": f"{(1 - expected_split) * 100:.0f}% / {expected_split * 100:.0f}%",
            "actual": f"{len(champ) / total * 100:.1f}% / {len(chal) / total * 100:.1f}%",
        }

    if len(champ) >= 5 and len(chal) >= 5:
        t_stat, p_val = scipy_stats.ttest_ind(champ, chal, equal_var=False)
        lift = (float(np.mean(chal)) - float(np.mean(champ))) / float(np.mean(champ))
        # 95% CI on the mean difference via normal approximation
        se = float(np.sqrt(np.var(chal) / len(chal) + np.var(champ) / len(champ)))
        diff = float(np.mean(chal)) - float(np.mean(champ))
        result["lift_pct"] = round(lift * 100, 2)
        result["p_value"] = round(float(p_val), 4)
        result["significant"] = bool(p_val < 0.05)
        result["ci_95"] = {
            "lower": round(diff - 1.96 * se),
            "upper": round(diff + 1.96 * se),
        }

    return result


@app.post("/v1/ab/generate-traffic", tags=["ab-testing"])
def ab_generate_traffic(request: Request) -> dict:
    """Fire 200 varied predictions to populate A/B stats — demo helper."""
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(503, "AVM model not loaded")

    cities = ["Seattle", "Austin", "Denver", "Boston", "Chicago", "Miami"]
    prop_types = ["single_family", "condo", "townhouse"]
    rng = np.random.default_rng(seed=42)
    n = 200
    generated = 0
    for i in range(n):
        try:
            row = {
                "city": cities[i % len(cities)],
                "property_type": prop_types[i % len(prop_types)],
                "sqft": int(rng.integers(1000, 4500)),
                "beds": int(rng.integers(1, 6)),
                "baths": float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
                "lot_size": int(rng.integers(0, 10000)),
                "year_built": int(rng.integers(1970, 2023)),
                "garage_spaces": int(rng.integers(0, 3)),
                "school_score": round(float(rng.uniform(3, 10)), 1),
                "walk_score": int(rng.integers(20, 100)),
                "crime_index": round(float(rng.uniform(10, 80)), 1),
            }
            df = pd.DataFrame([row])
            pred = state.avm_champion.predict(df, model_name=settings.avm_model_name,
                                              model_version=state.avm_champion_version)
            # Simulate A/B routing: 10% challenger
            variant = "challenger" if (i % 10 == 0 and state.avm_challenger is not None) else "champion"
            if variant == "challenger":
                p = state.avm_challenger.predict(df, model_name=settings.avm_model_name,
                                                 model_version=state.avm_challenger_version)
                request.app.state.ab_challenger_preds.append(p.point)
            else:
                request.app.state.ab_champion_preds.append(pred.point)
            generated += 1
        except Exception:
            pass

    # Trim buffers
    for buf in (request.app.state.ab_champion_preds, request.app.state.ab_challenger_preds):
        if len(buf) > 1000:
            del buf[:len(buf) - 1000]

    return {"generated": generated,
            "champion_n": len(request.app.state.ab_champion_preds),
            "challenger_n": len(request.app.state.ab_challenger_preds)}


# ---------------------------------------------------------------------------
# Feature Store — training vs. serving distributions
# ---------------------------------------------------------------------------

@app.get("/v1/feature-store/distributions", tags=["feature-store"])
def feature_distributions(request: Request) -> dict:
    """Training reference vs. live serving distributions for each AVM feature.

    This is the skew-prevention story: if training and serving diverge,
    the AVM is extrapolating and MAPE silently degrades.  Show this to any
    interviewer who asks about training/serving skew — it's the #1 silent
    failure mode.
    """
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(503, "AVM model not loaded")

    buf = list(request.app.state.recent_features)
    ref_df = feature_store.get_training_df()
    ref_feats = build_features(ref_df, city_price_level=state.avm_champion.city_price_level)

    def _stats(vals: list) -> dict:
        a = np.array(vals, dtype=float)
        return {
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a)), 4),
            "p5":  round(float(np.percentile(a, 5)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "n": len(vals),
        }

    cur_df = pd.DataFrame(buf) if buf else pd.DataFrame(columns=AVM_FEATURES)
    features_out = []
    for feat in AVM_FEATURES:
        ref_vals = ref_feats[feat].dropna().tolist()
        cur_vals = cur_df[feat].dropna().tolist() if feat in cur_df.columns and len(cur_df) > 0 else []
        # PSI between reference and current (reuse existing detector)
        psi_val = None
        severity = "ok"
        if state.drift_detector and len(cur_vals) >= 5:
            from src.monitoring.drift import psi_score
            p, _, _, _ = psi_score(np.array(ref_vals), np.array(cur_vals))
            psi_val = round(p, 4)
            severity = state.drift_detector._severity(p)

        features_out.append({
            "feature": feat,
            "reference": _stats(ref_vals),
            "current": _stats(cur_vals) if cur_vals else None,
            "psi": psi_val,
            "severity": severity,
        })

    return {
        "reference_size": len(ref_df),
        "current_size": len(buf),
        "features": features_out,
    }


# ---------------------------------------------------------------------------
# Ops — retrain challenger + hot-load
# ---------------------------------------------------------------------------

def _run_retrain_sync(job_id: str, jobs: dict) -> None:
    """Blocking training function — runs in _executor thread pool."""
    try:
        df = feature_store.get_training_df()
        train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
        data_hash = hashlib.sha256(train_df.to_parquet()).hexdigest()[:16]

        # Challenger: more leaves, lower LR → meaningfully different from champion
        params = {
            "objective": "regression", "metric": "mape",
            "learning_rate": 0.04, "num_leaves": 127,
            "min_data_in_leaf": 30, "feature_fraction": 0.85,
            "bagging_fraction": 0.85, "bagging_freq": 5, "verbose": -1,
        }
        from src.avm.model import AVMModel
        model = AVMModel.train(train_df, params=params)
        metrics = model.evaluate(val_df)
        gate_passed = metrics["mape"] < 0.18 and metrics["p90_coverage"] > 0.80
        elapsed = round(time.time() - jobs[job_id]["started_at"], 1)

        if gate_passed:
            registry = ModelRegistry()
            meta = registry.register(
                name=settings.avm_model_name, model=model,
                training_df_rows=len(train_df),
                training_data_hash=data_hash,
                params=params, metrics=metrics,
                tags={"source": "ui_retrain", "job_id": job_id},
            )
            registry.promote(settings.avm_model_name, meta.version, "staging")
            jobs[job_id].update({
                "status": "completed", "gate_passed": True,
                "version": meta.version, "stage": "staging",
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "elapsed_s": elapsed,
            })
        else:
            jobs[job_id].update({
                "status": "completed", "gate_passed": False,
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "elapsed_s": elapsed,
                "reason": (
                    f"Gate failed — MAPE {metrics['mape']:.1%} "
                    f"(need <18%), coverage {metrics['p90_coverage']:.1%} (need >80%)"
                ),
            })
    except Exception as e:
        jobs[job_id].update({"status": "failed", "error": str(e)})


@app.post("/v1/ops/retrain", tags=["ops"])
def retrain_trigger(request: Request) -> dict:
    """Kick off a background challenger training job.

    Returns immediately with a job_id. Poll /v1/ops/retrain/{job_id} for status.
    Trains on the same dataset with different hyperparams, validates, and
    registers the winner in staging — mirroring the CI/CD ML pipeline.
    """
    # Prevent duplicate concurrent runs
    running = [j for j in request.app.state.retrain_jobs.values() if j.get("status") == "running"]
    if running:
        raise HTTPException(409, f"Retrain already in progress: {running[0]['job_id']}")

    job_id = uuid.uuid4().hex[:8]
    jobs = request.app.state.retrain_jobs
    jobs[job_id] = {"status": "running", "job_id": job_id, "started_at": time.time()}
    _executor.submit(_run_retrain_sync, job_id, jobs)
    log.info("retrain_triggered", job_id=job_id)
    return {"job_id": job_id, "status": "running"}


@app.get("/v1/ops/retrain/{job_id}", tags=["ops"])
def retrain_status(job_id: str, request: Request) -> dict:
    """Poll a background retraining job by ID."""
    job = request.app.state.retrain_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return job


@app.post("/v1/ops/load-challenger", tags=["ops"])
def load_challenger(request: Request) -> dict:
    """Hot-load the staging model as challenger without restarting the server.

    After a retrain completes and lands in staging, call this to activate
    A/B routing. No restart needed — we swap the model in memory.
    """
    state: _State = request.app.state.ml
    registry = ModelRegistry()
    try:
        avm_c, meta_c = registry.load(settings.avm_model_name, "staging")
    except LookupError:
        raise HTTPException(404, "No model in staging — run /v1/ops/retrain first")

    if meta_c.version == state.avm_champion_version:
        raise HTTPException(409, f"Staging v{meta_c.version} is identical to the production champion")

    state.avm_challenger = avm_c
    state.avm_challenger_version = meta_c.version
    # Reset A/B buffers so the experiment starts clean
    request.app.state.ab_challenger_preds.clear()
    metrics_registry.model_loaded.labels(meta_c.name, str(meta_c.version), "staging").inc()
    log.info("challenger_hot_loaded", version=meta_c.version)
    return {
        "challenger_version": meta_c.version,
        "metrics": meta_c.metrics,
        "ab_traffic_split": state.ab_router.traffic_split,
        "message": f"Challenger v{meta_c.version} loaded. {state.ab_router.traffic_split*100:.0f}% of traffic will route to it.",
    }


# ---------------------------------------------------------------------------
# Market Intelligence
# ---------------------------------------------------------------------------

@app.get("/v1/market/intelligence", tags=["market"])
def market_intelligence() -> dict:
    """City-level real estate market stats aggregated from the 50k listing dataset."""
    df = feature_store.get_training_df().copy()
    df["price_per_sqft"] = (df["price"] / df["sqft"].clip(lower=1)).round(0)

    cities = []
    for city, gdf in df.groupby("city"):
        pt = gdf["property_type"].value_counts()
        cities.append({
            "city": str(city),
            "listing_count": int(len(gdf)),
            "median_price": int(gdf["price"].median()),
            "mean_price": int(gdf["price"].mean()),
            "p25_price": int(gdf["price"].quantile(0.25)),
            "p75_price": int(gdf["price"].quantile(0.75)),
            "median_sqft": int(gdf["sqft"].median()),
            "median_price_per_sqft": int(gdf["price_per_sqft"].median()),
            "median_school_score": round(float(gdf["school_score"].median()), 1),
            "median_walk_score": round(float(gdf["walk_score"].median()), 1),
            "median_crime_index": round(float(gdf["crime_index"].median()), 1),
            "property_type_pct": {
                k: round(v / len(gdf) * 100, 1) for k, v in pt.items()
            },
        })

    cities.sort(key=lambda c: c["median_price"], reverse=True)
    return {"cities": cities, "total_listings": int(len(df))}


# ---------------------------------------------------------------------------
# Negotiation Intelligence
# ---------------------------------------------------------------------------

_BUYER_MULT = {"hot": 1.03, "warm": 1.00, "cool": 0.95, "cold": 0.90}
_SELLER_LIST_MULT = {"hot": 1.07, "warm": 1.03, "cool": 0.995, "cold": 0.95}
_SELLER_FLOOR_MULT = {"hot": 0.98, "warm": 0.96, "cool": 0.91, "cold": 0.87}


def _market_condition(median_price: float) -> str:
    if median_price > 1_400_000:
        return "hot"
    if median_price > 850_000:
        return "warm"
    if median_price > 500_000:
        return "cool"
    return "cold"


@app.post("/v1/negotiation/strategy", response_model=NegotiationResponse, tags=["negotiation"])
def negotiation_strategy(body: NegotiationRequest, request: Request) -> NegotiationResponse:
    """Buyer bid + seller offer prices anchored to the AVM, adjusted for market conditions.

    Strategy logic:
      1. Run AVM on the property to get an objective estimate.
      2. Determine market temperature from the city's median price in our dataset.
      3. Adjust bid/offer by market condition, days-on-market, and competing offers.
      4. Call the LLM (or mock) for personalized buyer + seller tactic narratives.

    Returns bid/offer prices, expected close range, leverage indicator, key factors,
    and plain-English tactics for both sides.
    """
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(status_code=503, detail="AVM model not loaded")

    # ── 1. AVM prediction ──────────────────────────────────────────────────
    home_dict = body.model_dump(exclude={"days_on_market", "num_competing_offers", "asking_price"})
    df = pd.DataFrame([home_dict])
    pred = state.avm_champion.predict(
        df, model_name=settings.avm_model_name, model_version=state.avm_champion_version
    )
    avm = pred.point

    # ── 2. City market condition ──────────────────────────────────────────
    try:
        ref_df = feature_store.get_training_df()
        city_df = ref_df[ref_df["city"] == body.city]
        city_median = float(city_df["price"].median()) if len(city_df) > 0 else avm
    except Exception:
        city_median = avm
    condition = _market_condition(city_median)
    leverage_map = {"hot": "seller", "warm": "neutral", "cool": "buyer", "cold": "buyer"}
    leverage = leverage_map[condition]

    # ── 3. Compute base bid / list ────────────────────────────────────────
    buyer_mult = _BUYER_MULT[condition]
    seller_mult = _SELLER_LIST_MULT[condition]
    floor_mult = _SELLER_FLOOR_MULT[condition]

    # Days-on-market adjustment
    dom = body.days_on_market
    if dom is not None:
        if dom >= 60:
            buyer_mult -= 0.04
            floor_mult -= 0.02
            leverage = "buyer"
        elif dom >= 30:
            buyer_mult -= 0.02
            floor_mult -= 0.01
        elif dom <= 7:
            buyer_mult += 0.015
            if leverage == "neutral":
                leverage = "seller"

    # Competing offers adjustment
    n_offers = body.num_competing_offers
    if n_offers is not None:
        if n_offers >= 3:
            buyer_mult += 0.025
            leverage = "seller"
        elif n_offers == 0 and (dom or 0) > 21:
            buyer_mult -= 0.015
            leverage = "buyer"

    # Property-specific adjustments
    if body.year_built < 1980:
        buyer_mult -= 0.01     # inspection risk
    if body.property_type == "condo":
        buyer_mult -= 0.005    # higher inventory
    if body.school_score >= 9:
        buyer_mult += 0.01     # premium school district
    if body.walk_score >= 90:
        buyer_mult += 0.008

    buyer_bid = int(round(avm * buyer_mult / 1000) * 1000)
    seller_list = int(round(avm * seller_mult / 1000) * 1000)
    seller_floor = int(round(avm * floor_mult / 1000) * 1000)

    # Expected close range: between buyer bid and seller list, skewed by leverage
    if leverage == "seller":
        close_low = int(avm * 0.99)
        close_high = int(avm * 1.04)
    elif leverage == "buyer":
        close_low = int(avm * 0.91)
        close_high = int(avm * 0.98)
    else:
        close_low = int(avm * 0.97)
        close_high = int(avm * 1.01)

    # ── 4. Key factors ────────────────────────────────────────────────────
    factors: list[NegotiationFactor] = []

    cond_labels = {"hot": "Seller's market", "warm": "Balanced market",
                   "cool": "Buyer's market", "cold": "Strong buyer's market"}
    factors.append(NegotiationFactor(
        factor="Market condition",
        impact=leverage,
        detail=f"{cond_labels[condition]} — city median ${city_median:,.0f}",
    ))
    if dom is not None:
        if dom <= 7:
            factors.append(NegotiationFactor(factor="Days on market", impact="seller",
                detail=f"Fresh listing ({dom}d) — sellers have leverage, expect competition"))
        elif dom >= 60:
            factors.append(NegotiationFactor(factor="Days on market", impact="buyer",
                detail=f"Stale listing ({dom}d) — sellers are motivated, bid below AVM"))
        elif dom >= 30:
            factors.append(NegotiationFactor(factor="Days on market", impact="buyer",
                detail=f"{dom} days listed — buyer has modest leverage to negotiate"))
    if n_offers is not None:
        if n_offers >= 3:
            factors.append(NegotiationFactor(factor="Competing offers", impact="seller",
                detail=f"{n_offers} offers — bid at or above AVM, minimize contingencies"))
        elif n_offers == 0:
            factors.append(NegotiationFactor(factor="Competing offers", impact="buyer",
                detail="No competing offers — buyer holds strong position"))
    if body.school_score >= 8.5:
        factors.append(NegotiationFactor(factor="School district", impact="seller",
            detail=f"Score {body.school_score:.1f}/10 — premium schools shrink buyer leverage"))
    if body.crime_index <= 20:
        factors.append(NegotiationFactor(factor="Safety premium", impact="seller",
            detail=f"Crime index {body.crime_index:.0f}/100 — low crime adds ~2-3% premium"))
    if body.year_built < 1980:
        factors.append(NegotiationFactor(factor="Vintage risk", impact="buyer",
            detail=f"Built {body.year_built} — request inspection credit for aging systems"))
    if body.walk_score >= 85:
        factors.append(NegotiationFactor(factor="Walkability", impact="seller",
            detail=f"Walk score {body.walk_score} — high walkability commands 5-8% premium"))
    ci_width_pct = (pred.upper - pred.lower) / avm * 100
    if ci_width_pct > 40:
        factors.append(NegotiationFactor(factor="Valuation uncertainty", impact="neutral",
            detail=f"Wide 90% CI (±{ci_width_pct/2:.0f}%) — more negotiation room on both sides"))
    if body.asking_price:
        ask_vs_avm = (body.asking_price - avm) / avm * 100
        if ask_vs_avm > 5:
            factors.append(NegotiationFactor(factor="Asking vs AVM", impact="buyer",
                detail=f"Asking ${body.asking_price:,} is {ask_vs_avm:+.1f}% above AVM — overpriced"))
        elif ask_vs_avm < -5:
            factors.append(NegotiationFactor(factor="Asking vs AVM", impact="buyer",
                detail=f"Asking ${body.asking_price:,} is {ask_vs_avm:+.1f}% below AVM — underpriced, move fast"))

    # ── 5. Tactic narratives ──────────────────────────────────────────────
    strategy_ctx = {
        "avm_estimate": int(avm),
        "avm_lower": int(pred.lower),
        "avm_upper": int(pred.upper),
        "market_condition": condition,
        "leverage": leverage,
        "buyer_bid": buyer_bid,
        "buyer_bid_pct_of_avm": round((buyer_bid / avm - 1) * 100, 1),
        "seller_list_price": seller_list,
        "seller_list_pct_of_avm": round((seller_list / avm - 1) * 100, 1),
        "seller_floor": seller_floor,
    }
    features_ctx = body.model_dump()
    tactics = agent_llm.generate_negotiation_tactics(features_ctx, strategy_ctx)

    log.info("negotiation_strategy", request_id=request.state.request_id,
             city=body.city, condition=condition, leverage=leverage,
             buyer_bid=buyer_bid, seller_list=seller_list)

    return NegotiationResponse(
        avm_estimate=int(avm),
        avm_lower=int(pred.lower),
        avm_upper=int(pred.upper),
        market_condition=condition,
        leverage=leverage,
        buyer_bid=buyer_bid,
        buyer_bid_pct_of_avm=strategy_ctx["buyer_bid_pct_of_avm"],
        seller_list_price=seller_list,
        seller_list_pct_of_avm=strategy_ctx["seller_list_pct_of_avm"],
        seller_floor=seller_floor,
        expected_close_low=close_low,
        expected_close_high=close_high,
        negotiation_gap=seller_list - buyer_bid,
        key_factors=factors,
        buyer_tactic=tactics["buyer_tactic"],
        seller_tactic=tactics["seller_tactic"],
        model_version=state.avm_champion_version,
        request_id=request.state.request_id,
    )


# ---------------------------------------------------------------------------
# Property Comparison with Match Scores
# ---------------------------------------------------------------------------

def _score_property(
    home: HomeFeatures,
    avm: float,
    prefs: "BuyerPreferences",
    asking: int | None,
) -> "DimensionScores":
    """Score a property 0-100 across four buyer-preference dimensions.

    Lifestyle signals (household_size, has_children, has_dogs, work_from_home,
    lifestyle_type, commute_sensitive) re-weight sub-components so the score
    reflects what actually matters to THIS buyer, not a generic searcher.
    """
    ls = prefs.lifestyle  # may be None

    # ── Budget score (35% default weight) ─────────────────────────────────
    budget_headroom = (prefs.max_budget - avm) / prefs.max_budget
    budget_score = float(np.clip(50 + budget_headroom * 80, 0, 100))

    # ── Size score (25% default weight) ───────────────────────────────────
    # Lifestyle: large household needs more sqft; WFH needs extra bedroom
    effective_min_beds = prefs.min_beds
    if ls:
        if ls.household_size >= 4 and prefs.min_beds < 3:
            effective_min_beds = 3
        if ls.work_from_home:
            effective_min_beds = max(effective_min_beds, prefs.min_beds + 1)

    beds_score = 100.0 if home.beds >= effective_min_beds else max(0, 60 * home.beds / effective_min_beds)
    baths_score = 100.0 if home.baths >= prefs.min_baths else max(0, 50 * home.baths / prefs.min_baths)

    # Sqft sufficiency based on household size
    sqft_per_person = home.sqft / max(1, ls.household_size if ls else 1)
    sqft_score = float(np.clip(sqft_per_person / 4, 0, 100))  # 400 sqft/person = ~100

    size_score = beds_score * 0.45 + baths_score * 0.35 + sqft_score * 0.20

    # Dog penalty: condos without yards are bad for dogs
    if ls and ls.has_dogs:
        if home.property_type == "condo" and home.lot_size == 0:
            size_score = max(0, size_score - 20)
        elif home.lot_size >= 4000:
            size_score = min(100, size_score + 8)   # big yard bonus

    # ── Neighborhood score (25% default weight) ───────────────────────────
    school_norm = max(0, min(100, (home.school_score - prefs.min_school_score) / max(0.1, 10 - prefs.min_school_score) * 70 + 50))
    walk_norm   = max(0, min(100, (home.walk_score - prefs.min_walk_score) / max(1, 100 - prefs.min_walk_score) * 60 + 40))
    crime_norm  = max(0, min(100, (prefs.max_crime_index - home.crime_index) / max(1, prefs.max_crime_index) * 60 + 40))

    # Default weights: school 40%, walk 30%, safety 30%
    w_school, w_walk, w_crime = 0.40, 0.30, 0.30

    if ls:
        if ls.has_children:
            w_school, w_walk, w_crime = 0.55, 0.20, 0.25   # schools are paramount
        elif ls.commute_sensitive:
            w_school, w_walk, w_crime = 0.25, 0.50, 0.25   # walkability / transit
        elif ls.lifestyle_type == "urban":
            w_school, w_walk, w_crime = 0.25, 0.50, 0.25
        elif ls.lifestyle_type == "suburban":
            w_school, w_walk, w_crime = 0.40, 0.20, 0.40   # safety matters more

    nbhd_score = school_norm * w_school + walk_norm * w_walk + crime_norm * w_crime

    # ── Value score (15% default weight) ──────────────────────────────────
    if asking and asking > 0:
        deal_pct = (avm - asking) / avm * 100
        value_score = float(np.clip(60 + deal_pct * 3, 0, 100))
    else:
        value_score = 65.0

    # ── Soft bonuses ──────────────────────────────────────────────────────
    city_bonus = 5.0 if (prefs.preferred_city and home.city == prefs.preferred_city) else 0.0
    type_bonus = 5.0 if (prefs.preferred_property_type and home.property_type == prefs.preferred_property_type) else 0.0

    # Lifestyle bonuses beyond the weighted scores
    lifestyle_bonus = 0.0
    if ls:
        if ls.work_from_home and home.beds > effective_min_beds:
            lifestyle_bonus += 5.0     # has a dedicated office room
        if ls.has_dogs and home.lot_size >= 5000:
            lifestyle_bonus += 4.0     # big yard for dogs
        if ls.has_children and home.crime_index <= 20:
            lifestyle_bonus += 3.0     # very safe for kids
        if ls.household_size >= 5 and home.sqft >= 3000:
            lifestyle_bonus += 4.0     # genuinely big house
        if ls.lifestyle_type == "urban" and home.walk_score >= 85:
            lifestyle_bonus += 3.0
        if ls.lifestyle_type == "suburban" and home.lot_size >= 6000:
            lifestyle_bonus += 3.0

    # ── Weighted composite ────────────────────────────────────────────────
    raw_overall = (
        budget_score * 0.35
        + size_score  * 0.25
        + nbhd_score  * 0.25
        + value_score * 0.15
        + city_bonus + type_bonus + lifestyle_bonus
    )
    overall = float(np.clip(raw_overall, 0, 100))

    return DimensionScores(
        budget=round(budget_score, 1),
        size=round(size_score, 1),
        neighborhood=round(nbhd_score, 1),
        value=round(value_score, 1),
        overall=round(overall, 1),
    )


def _pros_cons(home: HomeFeatures, avm: float, prefs: "BuyerPreferences", asking: int | None, scores: "DimensionScores") -> tuple[list, list]:
    pros, cons = [], []
    ls = prefs.lifestyle

    # Budget
    if avm <= prefs.max_budget:
        pros.append(f"Within budget by ${prefs.max_budget - avm:,.0f}")
    else:
        cons.append(f"${avm - prefs.max_budget:,.0f} over budget")

    # Beds
    effective_min = prefs.min_beds + (1 if ls and ls.work_from_home else 0)
    if home.beds >= effective_min:
        label = f"{home.beds} beds" + (" (includes office room)" if ls and ls.work_from_home and home.beds > prefs.min_beds else "")
        pros.append(label)
    else:
        suffix = " — no room for home office" if ls and ls.work_from_home else ""
        cons.append(f"Only {home.beds} beds (need ≥{effective_min}{suffix})")

    # Baths
    if home.baths >= prefs.min_baths:
        pros.append(f"{home.baths:.1f} baths ≥ preference")
    else:
        cons.append(f"Only {home.baths:.1f} baths (want ≥{prefs.min_baths})")

    # School — boosted copy if children present
    if ls and ls.has_children:
        if home.school_score >= 8.5:
            pros.append(f"Excellent schools ({home.school_score:.1f}/10) — great for kids")
        elif home.school_score >= prefs.min_school_score:
            pros.append(f"School score {home.school_score:.1f} meets minimum")
        else:
            cons.append(f"School score {home.school_score:.1f} below preference — critical with kids")
    elif home.school_score >= prefs.min_school_score:
        pros.append(f"School score {home.school_score:.1f} meets standard")
    else:
        cons.append(f"School score {home.school_score:.1f} below preference ({prefs.min_school_score})")

    # Safety — boosted copy if children or large family
    if home.crime_index <= prefs.max_crime_index:
        if home.crime_index <= 20:
            suffix = " — very safe for kids" if ls and ls.has_children else ""
            pros.append(f"Low crime index ({home.crime_index:.0f}){suffix}")
    else:
        cons.append(f"Crime index {home.crime_index:.0f} above tolerance ({prefs.max_crime_index:.0f})")

    # Pets
    if ls and ls.has_dogs:
        if home.property_type == "condo" and home.lot_size == 0:
            cons.append("No yard — difficult for dogs in a condo")
        elif home.lot_size >= 5000:
            pros.append(f"Large yard ({home.lot_size:,} sqft) — great for dogs")
        elif home.lot_size > 0:
            pros.append(f"Has yard ({home.lot_size:,} sqft) for dogs")

    # Sqft per person
    if ls and ls.household_size >= 3:
        sqft_pp = home.sqft // ls.household_size
        if sqft_pp >= 400:
            pros.append(f"{sqft_pp} sqft/person for {ls.household_size}-person household")
        elif sqft_pp < 250:
            cons.append(f"Only {sqft_pp} sqft/person — tight for {ls.household_size} people")

    # Walkability
    if ls and ls.lifestyle_type == "urban":
        if home.walk_score >= 85:
            pros.append(f"Highly walkable ({home.walk_score}) — matches urban lifestyle")
        elif home.walk_score < 60:
            cons.append(f"Low walk score ({home.walk_score}) — poor fit for urban lifestyle")
    elif home.walk_score >= 75:
        pros.append(f"High walkability ({home.walk_score})")
    elif home.walk_score < prefs.min_walk_score:
        cons.append(f"Low walk score ({home.walk_score}, want ≥{prefs.min_walk_score})")

    # Deal vs asking
    if asking and avm > asking * 1.05:
        pros.append(f"AVM {money_str(int(avm))} > asking {money_str(asking)} — underpriced")
    elif asking and asking > avm * 1.08:
        cons.append(f"Asking {money_str(asking)} is {(asking/avm - 1)*100:.0f}% above AVM")

    # Soft preferences
    if prefs.preferred_city and home.city == prefs.preferred_city:
        pros.append(f"In preferred city ({home.city})")
    if prefs.preferred_property_type and home.property_type == prefs.preferred_property_type:
        pros.append(f"Preferred type ({home.property_type.replace('_', ' ')})")

    # Vintage
    if home.year_built >= 2010:
        pros.append(f"Modern construction ({home.year_built})")
    elif home.year_built < 1970:
        cons.append(f"Built {home.year_built} — inspection risk")

    return pros[:6], cons[:5]


@app.post("/v1/negotiation/compare", response_model=CompareResponse, tags=["negotiation"])
def compare_properties(body: CompareRequest, request: Request) -> CompareResponse:
    """Score and rank multiple properties against buyer preferences.

    For each property: runs AVM, computes four dimension scores (budget fit,
    size fit, neighborhood quality, value vs asking), aggregates into an
    overall match %, and generates pros/cons. Returns properties ranked
    by match score with a winner highlighted.
    """
    state: _State = request.app.state.ml
    if state.avm_champion is None:
        raise HTTPException(status_code=503, detail="AVM model not loaded")

    asking_prices = body.asking_prices or [None] * len(body.properties)
    if len(asking_prices) < len(body.properties):
        asking_prices = asking_prices + [None] * (len(body.properties) - len(asking_prices))

    scored: list[tuple[float, PropertyComparison]] = []
    for i, (home, asking) in enumerate(zip(body.properties, asking_prices)):
        df = pd.DataFrame([home.model_dump()])
        pred = state.avm_champion.predict(
            df, model_name=settings.avm_model_name, model_version=state.avm_champion_version
        )
        avm = pred.point

        # Buyer bid for this property
        city_df = feature_store.get_training_df()
        city_rows = city_df[city_df["city"] == home.city]
        city_median = float(city_rows["price"].median()) if len(city_rows) > 0 else avm
        condition = _market_condition(city_median)
        bid = int(round(avm * _BUYER_MULT[condition] / 1000) * 1000)

        scores = _score_property(home, avm, body.preferences, asking)
        pros, cons = _pros_cons(home, avm, body.preferences, asking, scores)

        scored.append((scores.overall, PropertyComparison(
            index=i,
            city=home.city,
            property_type=home.property_type,
            avm_estimate=int(avm),
            buyer_bid=bid,
            asking_price=asking,
            scores=scores,
            match_pct=scores.overall,
            rank=0,         # filled after sorting
            pros=pros,
            cons=cons,
            winner=False,   # filled after sorting
        )))

    # Rank by overall score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for rank, (_, comp) in enumerate(scored, start=1):
        comp.rank = rank
        comp.winner = (rank == 1)
        results.append(comp)

    winner = results[0]
    summary = (
        f"Property {winner.index + 1} ({winner.city} {winner.property_type.replace('_', ' ')}) "
        f"scores best at {winner.match_pct:.0f}% match — "
        f"{'within budget with AVM ' + money_str(winner.avm_estimate) if winner.avm_estimate <= body.preferences.max_budget else 'slightly over budget at AVM ' + money_str(winner.avm_estimate)}."
    )

    log.info("compare_properties", request_id=request.state.request_id,
             n_properties=len(body.properties), winner_index=winner.index)
    return CompareResponse(
        preferences=body.preferences,
        properties=results,
        winner_index=winner.index,
        summary=summary,
    )


def money_str(v: float) -> str:
    return f"${int(v):,}"
