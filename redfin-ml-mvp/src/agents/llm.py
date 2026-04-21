"""LLM client for AI agent features.

Uses Anthropic Claude (haiku — fast, cheap) when ANTHROPIC_API_KEY is set.
Falls back to mock responses so the demo runs without any API key.

This module is the single place where prompt caching and model selection live.
Both Hermes Agent and OpenClaw are model-agnostic and would call through an
equivalent abstraction — this stub mirrors that contract.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Mock responses — used when no API key is present so the demo still runs.
# ---------------------------------------------------------------------------
_MOCK_EXPLAIN = (
    "This {city} {property_type} is estimated at {price} primarily because "
    "it sits in a strong local market (city price tier is the top driver at {top_contrib}), "
    "compounded by {second} and {third}. "
    "The 90% confidence interval reflects typical valuation uncertainty for this home type."
)

_MOCK_TRIAGE = (
    "⚠️  {n_alarm} feature(s) are in ALARM state. "
    "The largest drift is in '{top_feature}' (PSI {max_psi:.3f}), "
    "which suggests the homes being queried recently differ significantly from the training distribution. "
    "Likely cause: seasonal demand shift or a new market segment entering the search funnel. "
    "Recommended action: collect labels on the drifted segment, retrain within 48 h, "
    "and consider narrowing the A/B challenger split until the new model is validated."
)

_CITIES = [
    "Seattle", "San Francisco", "Los Angeles", "Portland", "Austin", "Denver",
    "Chicago", "Boston", "Washington DC", "Atlanta", "Dallas", "Miami",
]

_PROPERTY_TYPES = {
    "condo": "condo", "condominium": "condo",
    "townhouse": "townhouse", "townhome": "townhouse",
    "multi.family": "multi_family", "multifamily": "multi_family",
    "single.family": "single_family", "craftsman": "single_family",
    "house": "single_family", "home": "single_family",
}


def _regex_parse_listing(text: str) -> dict:
    """Best-effort regex extraction when no API key is available."""
    import re
    t = text.lower()
    out: dict = {
        "city": "Seattle", "property_type": "single_family",
        "sqft": 1800, "beds": 3, "baths": 2.0,
        "lot_size": 4000, "year_built": 2000, "garage_spaces": 0,
        "school_score": 7.0, "walk_score": 65, "crime_index": 35.0,
    }
    # City
    for city in _CITIES:
        if city.lower() in t:
            out["city"] = city
            break
    # Property type
    for kw, pt in _PROPERTY_TYPES.items():
        if re.search(kw, t):
            out["property_type"] = pt
            break
    # Beds: "4BR", "4 bed", "4 bedroom"
    m = re.search(r'(\d+)\s*(?:br|bed(?:room)?s?)', t)
    if m:
        out["beds"] = int(m.group(1))
    # Baths
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ba|bath(?:room)?s?)', t)
    if m:
        out["baths"] = float(m.group(1))
    # Sqft
    m = re.search(r'([\d,]+)\s*(?:sq\.?\s*ft|sqft|square\s+feet)', t)
    if m:
        out["sqft"] = int(m.group(1).replace(",", ""))
    # Year built
    m = re.search(r'built\s+(?:in\s+)?(\d{4})', t)
    if m:
        out["year_built"] = int(m.group(1))
    # Garage
    m = re.search(r'(\d+)[\s-]car\s+garage', t)
    if m:
        out["garage_spaces"] = int(m.group(1))
    elif "garage" in t:
        out["garage_spaces"] = 1
    # Walk score
    m = re.search(r'walk\s+score\s+(\d+)', t)
    if m:
        out["walk_score"] = int(m.group(1))
    # School score
    m = re.search(r'school\s+score\s+(\d+(?:\.\d+)?)', t)
    if m:
        out["school_score"] = float(m.group(1))
    return out


def _regex_parse_search(query: str) -> dict:
    """Best-effort regex extraction for search queries."""
    import re
    t = query.lower()
    out: dict = {}
    # City
    for city in _CITIES:
        if city.lower() in t:
            out["city"] = city
            break
    # Price: "under $800k", "below $1.2 million", "max $500,000"
    m = re.search(r'(?:under|below|max|less than|up to)\s+\$?([\d,.]+)\s*(k|m|million|thousand)?', t)
    if m:
        val = float(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").lower()
        if suffix in ("k", "thousand"):
            val *= 1_000
        elif suffix in ("m", "million"):
            val *= 1_000_000
        out["max_price"] = int(val)
    # Min beds
    m = re.search(r'(\d+)\s*(?:br|bed(?:room)?s?)', t)
    if m:
        out["min_beds"] = int(m.group(1))
    # Property type
    for kw, pt in _PROPERTY_TYPES.items():
        if re.search(kw, t):
            out["property_type"] = pt
            break
    # School quality
    if any(w in t for w in ("school", "schools", "education")):
        out["min_school_score"] = 7.0
    return out


def _mock_explain(features: dict, prediction: dict) -> str:
    contribs = sorted(
        prediction.get("feature_contributions", {}).items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    labels = [k.replace("_", " ") for k, _ in contribs[:3]] + ["unknown"] * 3
    return _MOCK_EXPLAIN.format(
        city=features.get("city", "the"),
        property_type=features.get("property_type", "home").replace("_", " "),
        price=f"${prediction.get('point', 0):,.0f}",
        top_contrib=labels[0],
        second=labels[1],
        third=labels[2],
    )


def _mock_triage(drift_report: dict) -> str:
    features = drift_report.get("features", [])
    alarm_feats = [f for f in features if f.get("severity") == "alarm"]
    top = max(features, key=lambda f: f.get("psi", 0), default={})
    return _MOCK_TRIAGE.format(
        n_alarm=len(alarm_feats),
        top_feature=top.get("feature", "unknown"),
        max_psi=drift_report.get("max_psi", 0.0),
    )


# ---------------------------------------------------------------------------
# Real LLM call
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, max_tokens: int = 400) -> str:
    """Call Claude with prompt caching on the system prompt.

    If ANTHROPIC_API_KEY is unset, returns a mock so the demo runs offline.
    The `cache_control` block on the system prompt saves ~70% of tokens on
    repeated calls — important when the same system prompt is reused for
    every prediction explanation in a batch.
    """
    if not _API_KEY:
        # Return a sentinel so callers know to use their own mock
        return "__MOCK__"

    from anthropic import Anthropic  # lazy import — not needed at startup

    client = Anthropic(api_key=_API_KEY)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Public helpers — called by explainer.py
# ---------------------------------------------------------------------------

def explain_avm(features: dict, prediction: dict) -> str:
    system = (
        "You are a friendly real estate AI assistant helping home buyers understand "
        "property valuations. Write 2–3 plain-English sentences a buyer would understand — "
        "no ML jargon. Be specific about the top 2–3 price drivers."
    )
    contribs = sorted(
        prediction.get("feature_contributions", {}).items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]
    user = (
        f"Home: {json.dumps({k: features[k] for k in ['city','property_type','sqft','beds','baths','year_built'] if k in features})}\n"
        f"Predicted price: ${prediction.get('point', 0):,.0f}  "
        f"(90% CI: ${prediction.get('lower', 0):,.0f} – ${prediction.get('upper', 0):,.0f})\n"
        f"Top feature contributions (log-price):\n"
        + "\n".join(f"  {k}: {'+' if v >= 0 else ''}{v:.3f}" for k, v in contribs)
        + "\n\nExplain why this home is valued at approximately "
        f"${prediction.get('point', 0):,.0f}."
    )
    result = call_llm(system, user, max_tokens=200)
    return result if result != "__MOCK__" else _mock_explain(features, prediction)


def triage_drift(drift_report: dict) -> str:
    system = (
        "You are an ML operations engineer at a real estate company. "
        "Analyze this feature drift report and write 3–4 sentences covering: "
        "(1) which features are drifting, (2) likely real-world causes, "
        "(3) your recommended action. Be concise and actionable."
    )
    user = f"Drift report:\n{json.dumps(drift_report, indent=2)}"
    result = call_llm(system, user, max_tokens=300)
    return result if result != "__MOCK__" else _mock_triage(drift_report)


def parse_listing(text: str) -> dict:
    """Extract structured home features from raw MLS listing text."""
    system = (
        "Extract structured home features from MLS listing text. "
        "Return ONLY valid JSON with these exact keys: "
        "city (str), property_type (one of: single_family, condo, townhouse, multi_family), "
        "sqft (int), beds (int), baths (float), lot_size (int, 0 for condos), "
        "year_built (int), garage_spaces (int), "
        "school_score (float 1–10, estimate from context), "
        "walk_score (int 0–100, estimate from neighborhood description), "
        "crime_index (float 0–100, lower = safer, estimate). "
        "Use reasonable defaults for missing fields. Output JSON only, no prose."
    )
    user = f"Listing text:\n{text}"
    result = call_llm(system, user, max_tokens=300)
    if result == "__MOCK__":
        return _regex_parse_listing(text)
    import re
    m = re.search(r"\{.*\}", result, re.DOTALL)
    if not m:
        return _regex_parse_listing(text)
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return _regex_parse_listing(text)


def generate_negotiation_tactics(
    features: dict,
    strategy: dict,
) -> dict:
    """Generate buyer + seller negotiation tactics as plain-English strings.

    Returns {"buyer_tactic": str, "seller_tactic": str}.
    Falls back to rule-based strings when no API key is set.
    """
    system = (
        "You are an expert real estate negotiation coach. "
        "Given an AVM estimate, market condition, and property details, write "
        "two short paragraphs (2-3 sentences each) — one advising the BUYER on "
        "their bidding strategy (opening offer, contingencies, escalation), "
        "and one advising the SELLER on their pricing and concessions strategy. "
        "Be specific: mention the actual prices and percentages. No jargon."
    )
    user = (
        f"Property: {json.dumps({k: features[k] for k in ['city','property_type','sqft','beds','baths','year_built'] if k in features})}\n"
        f"AVM estimate: ${strategy['avm_estimate']:,}  (90% CI: ${strategy['avm_lower']:,}–${strategy['avm_upper']:,})\n"
        f"Market condition: {strategy['market_condition']} ({strategy['leverage']} leverage)\n"
        f"Recommended buyer bid: ${strategy['buyer_bid']:,} ({strategy['buyer_bid_pct_of_avm']:+.1f}% of AVM)\n"
        f"Recommended seller list: ${strategy['seller_list_price']:,} ({strategy['seller_list_pct_of_avm']:+.1f}% of AVM)\n"
        f"Days on market: {features.get('days_on_market', 'unknown')}\n"
        f"Competing offers: {features.get('num_competing_offers', 'unknown')}\n"
        "Write: BUYER TACTIC: <paragraph>\nSELLER TACTIC: <paragraph>"
    )
    result = call_llm(system, user, max_tokens=350)
    if result == "__MOCK__":
        return _mock_negotiation_tactics(features, strategy)

    buyer_tactic = ""
    seller_tactic = ""
    import re
    bm = re.search(r"BUYER TACTIC:\s*(.*?)(?:SELLER TACTIC:|$)", result, re.DOTALL | re.IGNORECASE)
    sm = re.search(r"SELLER TACTIC:\s*(.*?)$", result, re.DOTALL | re.IGNORECASE)
    if bm:
        buyer_tactic = bm.group(1).strip()
    if sm:
        seller_tactic = sm.group(1).strip()
    if not buyer_tactic or not seller_tactic:
        return _mock_negotiation_tactics(features, strategy)
    return {"buyer_tactic": buyer_tactic, "seller_tactic": seller_tactic}


def _mock_negotiation_tactics(features: dict, strategy: dict) -> dict:
    condition = strategy["market_condition"]
    leverage = strategy["leverage"]
    bid = strategy["buyer_bid"]
    list_p = strategy["seller_list_price"]
    avm = strategy["avm_estimate"]
    dom = features.get("days_on_market")
    offers = features.get("num_competing_offers")
    city = features.get("city", "this market")
    yr = features.get("year_built", 2000)

    # Buyer tactic
    if condition == "hot":
        b = (
            f"In {city}'s hot market, open at ${bid:,} — at or slightly above the AVM estimate "
            f"to stay competitive. Consider waiving the financing contingency if you're pre-approved, "
            f"and write a tight 5-day inspection window to signal seriousness. "
            f"An escalation clause up to ${int(avm * 1.06):,} protects you if competing offers surface."
        )
    elif condition == "warm":
        b = (
            f"This is a balanced {city} market — open at ${bid:,} (at AVM) and keep standard contingencies. "
            f"A 10-day inspection period is reasonable; sellers here typically expect near-ask offers. "
            f"You have modest room to negotiate $10k–$20k on inspection findings."
        )
    elif condition == "cool":
        dom_note = f" The {dom}-day listing history gives you leverage — " if dom and dom > 30 else " "
        b = (
            f"Open at ${bid:,} — about {abs(strategy['buyer_bid_pct_of_avm']):.0f}% below AVM.{dom_note}"
            f"Request a full inspection with credit for any deferred maintenance, and ask the seller to "
            f"cover 1–2% of closing costs. "
            f"{'The ' + str(yr) + ' vintage may reveal aging systems — price that in.' if yr < 1990 else 'There is room to negotiate up to $30k if the inspection reveals issues.'}"
        )
    else:  # cold
        b = (
            f"Open at ${bid:,} — well below AVM — and negotiate hard. "
            f"Request a full inspection with an allowance for repairs, plus seller-paid closing costs. "
            f"In {city}'s cold market, sellers often accept 8–12% below ask to close. "
            f"{'A ' + str(dom) + '-day DOM tells you they need to move — use it.' if dom and dom > 60 else 'Be patient; multiple counteroffers are normal here.'}"
        )

    # Seller tactic
    if condition == "hot":
        s = (
            f"List at ${list_p:,} — above AVM — to anchor expectations and invite competitive bids. "
            f"Hold offers for 5–7 days after listing to maximize competing-offer pressure. "
            f"{'With ' + str(offers) + ' known competing offers, counter at ask or above.' if offers and offers >= 2 else 'If no offers materialize in 10 days, drop to AVM and reassess.'}"
        )
    elif condition == "warm":
        s = (
            f"List at ${list_p:,} — just above AVM — giving you 2–3% of negotiating room. "
            f"Price the 'story' (school score {features.get('school_score', 7):.1f}/10, walkability) in your marketing. "
            f"If no offers in 14 days, reduce by ${int(avm * 0.02):,} and refresh the photos."
        )
    elif condition == "cool":
        s = (
            f"List at ${list_p:,} — at AVM — because overpricing in this market leads to extended DOM and stigma. "
            f"Offer a home warranty ($400–600) and pre-pay 6 months of HOA if applicable to differentiate. "
            f"Budget for 1–2% in buyer-requested credits to close the deal."
        )
    else:  # cold
        s = (
            f"Price at ${list_p:,} and be ready to negotiate down to ${strategy['seller_floor']:,}. "
            f"Consider a price reduction every 21 days if there are no showings. "
            f"Offering seller-paid rate buydown (1 point ≈ ${int(avm * 0.01):,}) can unlock buyers who can't qualify at full rate."
        )

    return {"buyer_tactic": b, "seller_tactic": s}


def parse_search_query(query: str) -> dict:
    """Convert a natural-language home search query into structured filter params."""
    system = (
        "Parse a natural-language home search query into structured JSON filters. "
        "Return ONLY valid JSON with these optional keys (omit keys not mentioned): "
        "city (str), max_price (int), min_price (int), "
        "min_beds (int), max_beds (int), min_baths (float), "
        "min_sqft (int), max_sqft (int), "
        "property_type (one of: single_family, condo, townhouse, multi_family), "
        "min_school_score (float 1–10). "
        "Infer price from phrases like 'under $800k' or 'around $1.2 million'. "
        "Output JSON only."
    )
    user = f"Search query: {query}"
    result = call_llm(system, user, max_tokens=200)
    if result == "__MOCK__":
        return _regex_parse_search(query)
    import re
    m = re.search(r"\{.*\}", result, re.DOTALL)
    if not m:
        return _regex_parse_search(query)
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return _regex_parse_search(query)
