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


def market_intelligence_agent(
    city: str,
    property_type: Optional[str],
    city_stats: dict,
    context_notes: str,
) -> dict:
    """Generate multi-factor market intelligence: seasonal, development, regulatory signals."""
    import datetime
    month = datetime.datetime.now().month
    month_name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1]

    system = (
        "You are a senior real estate market analyst. Given city statistics and contextual signals, "
        "provide a comprehensive market intelligence report covering:\n"
        "1. Seasonal price patterns for the current month\n"
        "2. New construction activity and supply effects\n"
        "3. Infrastructure & commercial development signals (transit, retail, office parks)\n"
        "4. Regulatory and macroeconomic factors (zoning, interest rates, tax law)\n"
        "5. Near-term price trajectory (3–6 month outlook)\n\n"
        "Format EXACTLY as:\n"
        "NARRATIVE: <2–3 paragraphs>\n"
        "PRICE_TREND: rising|stable|declining\n"
        "BUY_TIMING: <one sentence>\n"
        "SELL_TIMING: <one sentence>\n"
        "SIGNALS:\n"
        "- CATEGORY: seasonal | SIGNAL: <text> | IMPACT: positive|negative|neutral | CONFIDENCE: high|medium|low\n"
        "(list 5–7 signals covering different categories)"
    )
    user = (
        f"City: {city}\n"
        f"Property focus: {property_type or 'all types'}\n"
        f"Current month: {month} ({month_name})\n"
        f"City statistics (from 50k listings dataset):\n{json.dumps(city_stats, indent=2)}\n"
        f"User context / known signals: {context_notes or 'none provided'}\n\n"
        "Generate the market intelligence report."
    )
    result = call_llm(system, user, max_tokens=700)
    if result == "__MOCK__":
        return _mock_market_intelligence(city, city_stats, context_notes, month)

    import re
    narrative = ""
    price_trend = "stable"
    best_time_to_buy = "Spring (March–May) typically offers the widest inventory selection."
    best_time_to_sell = "List in March–April to capture peak spring buyer demand."
    signals = []

    nm = re.search(r"NARRATIVE:\s*(.*?)(?:PRICE_TREND:|$)", result, re.DOTALL | re.IGNORECASE)
    if nm:
        narrative = nm.group(1).strip()
    tm = re.search(r"PRICE_TREND:\s*(\w+)", result, re.IGNORECASE)
    if tm and tm.group(1).lower() in ("rising", "stable", "declining"):
        price_trend = tm.group(1).lower()
    bm = re.search(r"BUY_TIMING:\s*(.*?)(?:\n|SELL_TIMING:)", result, re.DOTALL | re.IGNORECASE)
    if bm:
        best_time_to_buy = bm.group(1).strip()
    sm = re.search(r"SELL_TIMING:\s*(.*?)(?:\n|SIGNALS:)", result, re.DOTALL | re.IGNORECASE)
    if sm:
        best_time_to_sell = sm.group(1).strip()
    for m in re.finditer(
        r"CATEGORY:\s*(\w+)\s*\|\s*SIGNAL:\s*(.*?)\s*\|\s*IMPACT:\s*(\w+)\s*\|\s*CONFIDENCE:\s*(\w+)",
        result, re.IGNORECASE,
    ):
        cat = m.group(1).lower()
        imp = m.group(3).lower()
        conf = m.group(4).lower()
        signals.append({
            "category": cat,
            "signal": m.group(2).strip(),
            "impact": imp if imp in ("positive", "negative", "neutral") else "neutral",
            "confidence": conf if conf in ("high", "medium", "low") else "medium",
        })

    if not narrative:
        return _mock_market_intelligence(city, city_stats, context_notes, month)

    return {
        "narrative": narrative,
        "signals": signals,
        "price_trend": price_trend,
        "best_time_to_buy": best_time_to_buy,
        "best_time_to_sell": best_time_to_sell,
    }


def _mock_market_intelligence(city: str, city_stats: dict, context_notes: str, month: int) -> dict:
    """Rule-based fallback when no API key is available."""
    median = city_stats.get("median_price", 750_000)
    recent_construction_pct = city_stats.get("recent_construction_pct", 15.0)
    median_year = city_stats.get("median_year_built", 1995)
    median_school = city_stats.get("median_school_score", 7.0)
    listing_count = city_stats.get("listing_count", 5000)

    # Market tier
    if median > 1_400_000:
        tier = "premium"
        trend = "rising"
        trend_note = f"{city} remains a premium market with constrained inventory. Prices continue to appreciate as demand from tech and finance sectors outpaces supply."
    elif median > 850_000:
        tier = "high"
        trend = "stable"
        trend_note = f"{city}'s market is normalizing after post-pandemic appreciation. Prices have stabilized with modest year-over-year growth of 2–4%."
    elif median > 500_000:
        tier = "mid"
        trend = "stable"
        trend_note = f"{city} offers relative affordability, attracting relocation demand. Supply is balanced but new construction is adding pressure in the mid range."
    else:
        tier = "value"
        trend = "declining"
        trend_note = f"{city} is experiencing softening demand as buyers have more options and mortgage rates remain elevated. Seller concessions are becoming common."

    # Seasonal signals
    spring = month in (3, 4, 5)
    summer = month in (6, 7, 8)
    fall   = month in (9, 10, 11)
    if spring:
        season_signal = "Peak spring listing season — inventory up 15–25%. More choices for buyers but also more competition."
        season_impact = "positive"
        buy_time = "Now (spring) is ideal for selection — inventory is at its annual peak."
        sell_time = "List immediately — spring demand typically closes 8–12% faster than fall listings."
    elif summer:
        season_signal = "Summer market: slightly lower inventory as families pause. Back-to-school relocation demand peaks in July."
        season_impact = "neutral"
        buy_time = "Late summer (August) often sees price reductions as overpriced listings age — good negotiating position."
        sell_time = "Price competitively — summer buyers are motivated but supply is still adequate."
    elif fall:
        season_signal = "Fall slowdown begins in October. Listing volume drops 20–30%, but serious buyers remain active."
        season_impact = "neutral"
        buy_time = "Fall offers reduced competition — you may be the only offer on the table."
        sell_time = "List by mid-September to catch fall buyers before the holiday freeze."
    else:  # winter
        season_signal = "Winter market: lowest inventory of the year. Fewer listings but motivated sellers."
        season_impact = "negative"
        buy_time = "Winter buyers face limited choices but can negotiate more — sellers who list in Jan/Feb are motivated."
        sell_time = "Avoid listing Dec–Jan unless urgent; spring lists typically command 3–5% premium."

    construction_note = (
        f"{recent_construction_pct:.0f}% of {city} listings are post-2015 construction, "
        + ("indicating active new supply that caps appreciation in the mid-price segment." if recent_construction_pct > 20
           else "showing limited new supply — older housing stock supports tighter inventory.")
    )

    context_signal = None
    if context_notes:
        context_signal = {
            "category": "development",
            "signal": f"User-reported signal: {context_notes}. This type of development typically increases nearby values 3–8% within 18 months.",
            "impact": "positive",
            "confidence": "medium",
        }

    inventory_desc = "a tight market by historical standards" if listing_count < 3000 else "adequate for current demand levels"
    school_impact = "homes near top-rated schools command a 10–15% premium" if median_school >= 7.5 else "mid-rated schools have less price impact"
    macro_cta = "Watch for zoning reform discussions in high-cost cities that could unlock ADU and density opportunities." if tier in ("premium", "high") else "Value-tier markets are most sensitive to rate changes — any Fed cut could quickly re-inflate demand."
    vintage_note = "aging stock — kitchen/bath updates add 5–8% value" if median_year < 1990 else "relatively modern inventory with lower deferred maintenance risk"

    narrative = (
        f"{trend_note}\n\n"
        f"Supply dynamics: {construction_note} The city has {listing_count:,} active listings, "
        f"{inventory_desc}. "
        f"School quality (median {median_school:.1f}/10) is a sustained demand driver — {school_impact}.\n\n"
        f"Macro context: Interest rates above 6.5% continue to suppress affordability and reduce transaction volume. "
        f"Buyers who can secure below-market rates or pay cash have significant negotiating advantage. "
        f"{macro_cta}"
    )

    signals = [
        {"category": "seasonal", "signal": season_signal, "impact": season_impact, "confidence": "high"},
        {"category": "supply", "signal": construction_note, "impact": "negative" if recent_construction_pct > 20 else "positive", "confidence": "high"},
        {"category": "demand", "signal": f"School quality (avg {median_school:.1f}/10) sustains buyer demand for family homes near top districts.", "impact": "positive", "confidence": "high"},
        {"category": "regulatory", "signal": "Fed rate policy: elevated mortgage rates (6.5–7%) constrain buyer pool, extending average DOM.", "impact": "negative", "confidence": "high"},
        {"category": "development", "signal": f"Median home vintage ({int(median_year)}) suggests {vintage_note}.", "impact": "neutral", "confidence": "medium"},
        {"category": "demand", "signal": f"Remote work flexibility continues to drive {city} relocation demand from higher-cost coastal markets.", "impact": "positive", "confidence": "medium"},
    ]
    if context_signal:
        signals.insert(1, context_signal)

    return {
        "narrative": narrative,
        "signals": signals,
        "price_trend": trend,
        "best_time_to_buy": buy_time,
        "best_time_to_sell": sell_time,
    }


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
