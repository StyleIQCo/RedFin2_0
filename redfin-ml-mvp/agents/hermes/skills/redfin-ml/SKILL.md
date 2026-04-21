---
name: redfin-ml
description: >
  Use Redfin's ML platform to estimate home prices, find similar listings,
  run natural-language searches, generate valuation narratives, and triage
  model drift. Covers the full AVM + recommender lifecycle.
---

# Redfin ML Skill

This skill gives you access to Redfin's production ML platform:
- **AVM** (Automated Valuation Model) — LightGBM + quantile twins, 11% MAPE
- **Home Recommender** — content-based ANN, drives 27% of platform traffic
- **AI Agent endpoints** — LLM-powered explanation, drift triage, listing parsing, NL search

API base URL: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

---

## Tool: nl_search

Natural-language home search. Convert the user's query into filters and return
ranked results.

```
POST /v1/agent/nl-search
{
  "query": "3BR under $800k near good schools in Seattle",
  "limit": 10
}
```

Returns: `parsed_filters` (what was inferred), `results` (listing list),
`result_count`.

**When to use:** user asks to find/search for homes, e.g.
"Find me a condo in Austin under $600k", "What 4BR single-family homes are available in Denver?"

---

## Tool: price_home

Estimate a home's value.

```
POST /v1/avm/predict
{
  "city": "Seattle", "property_type": "single_family",
  "sqft": 2200, "beds": 3, "baths": 2.5, "lot_size": 5000,
  "year_built": 2005, "garage_spaces": 2,
  "school_score": 8.5, "walk_score": 72, "crime_index": 25
}
```

Returns: `point` (USD), `lower`/`upper` (90% CI), `feature_contributions`,
`model_name`, `model_version`, `variant` (champion/challenger).

**When to use:** user provides home details and asks for a valuation.

---

## Tool: explain_valuation

Generate a plain-English explanation of an AVM result.

```
POST /v1/agent/explain
{
  "features": { <HomeFeatures> },
  "prediction": { <PricePrediction from price_home> }
}
```

Returns: `narrative` — 2–3 sentences a buyer would understand.

**When to use:** always call after `price_home` to give the user context,
not just a number.

---

## Tool: parse_listing

Extract features from raw MLS text and get an instant valuation + narrative
in a single call.

```
POST /v1/agent/parse-listing
{
  "text": "Charming 3BR/2BA craftsman in Queen Anne, Seattle. 1,850 sqft,
           updated kitchen, 2-car garage. Built 2001. Walk score 85."
}
```

Returns: `features` (extracted), `prediction` (AVM), `narrative` (plain English).

**When to use:** user pastes a listing description and wants an instant price.
This is the "listing intelligence" flow — unstructured → structured → priced.

---

## Tool: similar_homes

Find similar listings with explainability.

```
POST /v1/recommender/similar
{
  "listing_id": 42,
  "k": 8,
  "same_city": true
}
```

Returns: list of similar homes with `reasons` (e.g. "similar size · same bedroom count").

**When to use:** user wants alternatives to a specific listing.

---

## Tool: triage_drift

Check model health and get an LLM narrative explaining any drift.

```
# Step 1 — raw drift data (needs ≥30 recent predictions in the ring buffer)
GET /v1/drift/report

# Step 2 — LLM triage narrative
POST /v1/agent/triage
```

Returns: `overall_severity` (ok/warn/alarm), `max_psi_feature`, `narrative`.

**When to use:** when asked about model health, after a monitoring alert, or on
a scheduled triage cadence (e.g. hourly cron).

---

## Workflow patterns

### Pattern A — Full search-to-explanation flow
1. `nl_search` → show table of results
2. User picks a listing → `similar_homes` for alternatives
3. User asks "how much is #42 worth?" → `price_home` → `explain_valuation`

### Pattern B — Listing intelligence
1. User pastes MLS text → `parse_listing` (single call: parse + price + explain)
2. User asks "what's similar?" → `similar_homes` with the returned listing_id

### Pattern C — Scheduled model health check (cron)
1. `triage_drift` → if severity is alarm, alert the on-call engineer
2. Log the narrative to `~/.hermes/logs/redfin-drift-<date>.md` for audit trail

### Pattern D — Self-improvement loop
After completing any of the above patterns, analyze the steps taken and
consider whether a reusable sub-skill should be saved (e.g. a "price-and-explain"
combined shortcut). Save to `~/.hermes/workspace/skills/redfin-ml/` if it
reduces future token usage.

---

## Notes

- All endpoints return a `request_id` for tracing through logs.
- Price estimates use synthetic training data; real deployment would use live MLS feeds.
- If `ANTHROPIC_API_KEY` is not set, agent endpoints return sensible mock narratives
  so the demo still works offline.
- Model version is always included in responses — useful for debugging if a new
  model was deployed between requests.
