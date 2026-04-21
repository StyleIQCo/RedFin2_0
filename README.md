# Redfin ML MVP

**A production-shaped ML platform for real estate — built as an interview artifact for the Senior Machine Learning Engineer role on Redfin's Applied Machine Learning team.**

This MVP demonstrates how I would approach the two core problems the AML team owns:

1. **Automated Valuation Model (AVM)** — "How much is this home worth?"
2. **Home Recommender** — "Where should I live?" / "What other homes should I see?"

...wrapped in the MLOps stack the JD calls out: containerized serving, model registry + versioning, drift detection, A/B testing, Prometheus metrics, CI/CD pipeline, and a live demo UI.

---

## Why this shape

The JD emphasizes **bridging research and production**: converting prototypes into performant, maintainable systems, building automated MLOps pipelines, and owning model health in production. So this repo is deliberately **infra-heavy, model-lightweight** — the models themselves are small (LightGBM for AVM, cosine similarity for recommender) because in a real interview conversation I'd rather walk through *how* we'd keep a GBDT-ensemble AVM healthy at national scale than debate hyperparameters.

---

## What's inside

```
redfin-ml-mvp/
├── src/
│   ├── avm/              # Automated Valuation Model (LightGBM)
│   ├── recommender/      # Home recommender (content-based + collaborative stub)
│   ├── serving/          # FastAPI inference service + A/B router + feature store
│   ├── registry/         # Model registry with versioning, lineage, rollback
│   └── monitoring/       # PSI drift, Prometheus metrics, structured logging
├── tests/                # pytest unit + integration tests
├── data/                 # Synthetic real estate data generator
├── k8s/                  # Deployment, HPA, ServiceMonitor, ConfigMap
├── .github/workflows/    # CI/CD for ML (train → validate → package → deploy gate)
├── Dockerfile
├── docker-compose.yml    # Local stack: API + Prometheus + Grafana
└── ui/index.html         # Live demo web UI

DESIGN.md                 # Full system design + tradeoffs
Redfin-ML-MVP-Deck.pptx   # 10-slide architecture deck
```

---

## Quick start

```bash
# 1. Install + generate synthetic data + train both models
cd redfin-ml-mvp
pip install -r requirements.txt
python -m data.generate                     # creates data/homes.parquet
python -m src.avm.train                     # registers AVM v1
python -m src.recommender.train             # registers recommender v1

# 2. Run the API
uvicorn src.serving.app:app --reload
# → http://localhost:8000/docs

# 3. Open the demo UI
open ../ui/index.html
```

Or just `docker-compose up` — brings the API + Prometheus + Grafana up together.

---

## The interview story

This is structured around **three demos** I can walk through:

1. **"Here's a home" → price prediction** (AVM in action, with confidence interval)
2. **"Here's my shortlist" → similar homes** (recommender in action)
3. **"Here's what happens when data drifts"** — run the drift simulator, watch the drift detector flip from green to red and the alert fire

Each demo maps to a JD bullet:
- *Productionize models* → the whole serving stack
- *MLOps best practices* → `.github/workflows/ml-cicd.yml` + model registry
- *Optimize for inference* → feature store with hot cache + batch prediction endpoint
- *Monitor in production* → `monitoring/drift.py` + `monitoring/metrics.py`
- *Data drift / concept drift* → `tests/test_drift.py` simulates both
- *A/B testing* → `serving/ab_testing.py` with deterministic hash-based bucketing

See `DESIGN.md` for the full write-up.
