# Redfin Applied ML — Design Doc

> Written as a pre-interview artifact for the Senior Machine Learning Engineer role on the Applied Machine Learning team. Paired with a runnable MVP (`redfin-ml-mvp/`) and a 10-slide deck (`Redfin-ML-MVP-Deck.pptx`).

---

## 1. Problem framing

The Applied ML team owns two customer-facing systems that together sit at the top of the Redfin funnel:

1. **Automated Valuation Model (AVM)** — answers *"How much is this home worth?"*. Runs on every listing page, powers pricing recommendations, and feeds downstream products like the Redfin Estimate and the broker CMA.
2. **Home Recommender / Brokerage Recommendations** — answers *"Where should I live?"* and *"What other homes should I see?"*. Per the JD, this drives **27% of traffic** to Redfin platforms.

Both are served at national scale to hundreds of millions of users, across multi-modal inputs (structured property data, images, 3D scans, documents, text).

The hardest thing about these systems is not the ML — it's keeping them alive, fresh, and trustworthy in production. **Most AVM and recommender regressions are silent**: a feature pipeline changes its imputation default, a market launches, an upstream parser drops a column. Users keep getting answers; the answers are just subtly wrong. That's the problem this MVP is designed around.

---

## 2. What I built

A runnable MVP that shows the shape of how I'd build + operate these systems at Redfin. It's deliberately infra-heavy and model-lightweight: the models are small (LightGBM AVM with quantile CIs, ANN-based content recommender) because I'd rather spend interview time on how we keep them healthy than on hyperparameter choice.

### Components

```
redfin-ml-mvp/
├── src/avm/                     LightGBM regressor + quantile twins for 90% CI
├── src/recommender/             Content-based ANN w/ explainable recs
├── src/registry/                Filesystem model registry (versioning + stages + rollback)
├── src/serving/                 FastAPI app, A/B router, feature-store abstraction
├── src/monitoring/              PSI drift detector, Prometheus metrics, structured logging
├── tests/                       Training/serving skew, drift, A/B, registry, full E2E
├── k8s/                         Deployment, HPA, PDB, ServiceMonitor, ConfigMap
├── .github/workflows/           CI/CD for ML: test → train → validate → shadow → promote
├── ops/                         Prometheus config + alert rules + Grafana datasource
└── Dockerfile, docker-compose.yml
```

---

## 3. System architecture

```
                       ┌──────────────────────────────────────────────────────┐
                       │                 Redfin ML Serving Plane              │
                       │                                                      │
  Clients ─────▶ Ingress (L7 LB) ──▶ FastAPI (k8s Deployment, 3..30 replicas) │
                       │                   │                                  │
                       │                   ├─▶ A/B Router ─▶ Model Cache      │
                       │                   │    (champion/challenger/shadow)   │
                       │                   │                                  │
                       │                   ├─▶ Feature Store (online)         │
                       │                   │      │                           │
                       │                   │      └─▶ Redis (hot cache, TTL)  │
                       │                   │                                  │
                       │                   └─▶ Monitoring:                    │
                       │                        • Prometheus (/metrics)       │
                       │                        • structlog → Datadog         │
                       │                        • Drift detector (PSI ring)   │
                       └──────────────────────────────────────────────────────┘

                       ┌──────────────────────────────────────────────────────┐
                       │                 Offline / Training Plane             │
                       │                                                      │
  Warehouse ─▶ Spark feature pipelines ─▶ Offline store (Parquet/Iceberg)     │
                       │                                      │               │
                       │        Training DAG (Airflow) ◀──────┘               │
                       │              │                                       │
                       │              ▼                                       │
                       │        LightGBM train ─▶ Offline eval gate ─▶ Model  │
                       │                                              Registry│
                       └──────────────────────────────────────────────────────┘

                       ┌──────────────────────────────────────────────────────┐
                       │                    MLOps Plane                       │
                       │                                                      │
  CI (GH Actions) ─▶ unit → feature skew → train → eval gate → build image ─▶│
                       │                                                      │
                       │            ─▶ Helm/Argo deploy → staging (shadow)   │
                       │                                   ↓                  │
                       │                            Protected env gate        │
                       │                                   ↓                  │
                       │                            Production (A/B)          │
                       └──────────────────────────────────────────────────────┘
```

---

## 4. Where the JD requirements land in the code

| JD requirement | Where it lives in the repo |
|---|---|
| Productionize models, research → prod | `src/avm/model.py`, `src/recommender/model.py`, `src/serving/app.py` |
| MLOps: CI/CD, retraining, versioning | `.github/workflows/ml-cicd.yml`, `src/registry/model_registry.py` |
| Inference optimization | `src/serving/feature_store.py` (online cache, batch endpoint), `src/avm/model.py` (quantile booster share data set) |
| Monitor models, drift, latency | `src/monitoring/drift.py` (PSI), `src/monitoring/metrics.py` (Prometheus), `ops/rules.yml` (alerts) |
| AVM + recommender iteration | `src/avm/`, `src/recommender/` + `tests/test_features.py` for skew protection |
| Technical bridge / stakeholder enablement | The explainability story: `feature_contributions` in AVM responses + `reasons` in recommender responses |
| Python production-grade | `src/`, typed, structured logging, pydantic schemas, tested |
| PyTorch/TF/sklearn competency | sklearn (recommender) + LightGBM (AVM) — chosen over DL because GBDT remains SOTA for tabular real-estate regression; DL would be reserved for multi-modal (images, 3D) components |
| Docker / Kubernetes | `Dockerfile` (multi-stage), `k8s/*.yaml` (Deployment, HPA, PDB, ServiceMonitor) |
| A/B testing | `src/serving/ab_testing.py` + `tests/test_ab_router.py` |
| SQL / distributed data | Design-doc section 8 below (not in the MVP code, but how I'd build it) |

---

## 5. Model lifecycle

This is the full loop — dotted arrows are automated, solid arrows are human-in-the-loop.

```
  (new data arrives)              (on-call / PR approver)
        │                                    ▲
        ▼                                    │
  ┌─────────────┐   ┌──────────┐  ┌──────────┐   ┌────────────┐   ┌────────────┐
  │ Feature ETL │──▶│ Training │─▶│  Offline │──▶│   Registry │──▶│  Promotion │
  │   (Spark)   │   │   (DAG)  │  │   Eval   │   │ (candidate │   │     gate   │
  └─────────────┘   └──────────┘  │   gate   │   │  → staging)│   │ (prod)     │
                                   └──────────┘   └────────────┘   └──────┬─────┘
                                                                          │
           ┌──────────────────────────────────────────────────────────────┘
           ▼
    ┌──────────┐      ┌──────────────┐      ┌──────────────┐
    │  Shadow  │─────▶│ A/B champion │─────▶│  Production  │
    │  deploy  │      │ vs challenger│      │  champion    │
    └─────┬────┘      └──────┬───────┘      └──────┬───────┘
          │                  │                     │
          ▼                  ▼                     ▼
       predictions         metrics              metrics + drift ──▶ if alarm → rollback
       compared,           compared,                                             │
       no user risk        stat test                                             ▼
                                                                         previous prod model
                                                                         promoted automatically
```

Two promotions require approval:
1. **Candidate → Production.** The offline eval gate decides candidate viability; a human decides production-readiness. This is load-bearing — the cost of a bad AVM model in production is measured in customer trust, not just revenue.
2. **Rollback.** Automated on `AVMDataDriftAlarm` + `AVMErrorRate` alert triggers (rule: > 1% error for 5m + PSI ≥ 0.25 for 15m).

---

## 6. Monitoring strategy

Four signals, four thresholds, one response.

| Signal | Metric | Warn | Alarm | Primary runbook action |
|---|---|---|---|---|
| **Latency** | `redfin_ml_request_latency_seconds` p95 | > 150ms | > 250ms | Check HPA headroom, investigate slow feature-store queries |
| **Error rate** | `redfin_ml_requests_total{status="error"}` / total | > 0.1% | > 1% | Check logs, recent deploys; rollback if correlated |
| **Input drift** | `redfin_ml_drift_max_psi` (PSI) | ≥ 0.10 | ≥ 0.25 | Compare feature histograms; check upstream pipelines for schema drift |
| **Output drift** | `redfin_ml_avm_prediction_usd` median shift vs. 24h | 1.2x | 1.5x | Rare but high-priority — almost always means a model-level issue |

All four are in `ops/rules.yml`. The PSI metric updates on every `/v1/drift/report` call; in prod we'd have a sidecar cron job querying this every minute.

### Why PSI specifically
PSI handles mixed distributions gracefully (continuous + categorical), has established industry thresholds (0.10/0.25), is easy to interpret (bin-level contribution shows *which* bin is drifting), and is cheap to compute incrementally. The alternative — KS test — is great for continuous features but awkward for categoricals and sensitive to sample size. Combining both in a real prod deploy is reasonable; for this MVP, PSI is the primary with KS as a secondary check behind it.

---

## 7. Training / serving skew — the thing I care most about

In my experience, the #1 cause of silent AVM regressions is training/serving skew. Two subtle variants:

1. **Transform skew.** The code that computes features in training ≠ the code in serving. Fix: the `build_features()` function in `src/avm/features.py` is imported by both the training entrypoint and the FastAPI app. `tests/test_features.py` directly tests that training and serving produce identical outputs for identical inputs.
2. **Distribution skew.** The feature arrives at serving-time from a different source than training-time (e.g. real-time vs nightly ETL), so the values differ subtly. Fix: the feature-store abstraction in `src/serving/feature_store.py` exposes the *same interface* for offline and online retrieval; drift detection is the backstop that catches any residual skew.

The third variant — **label leakage** — is handled in training code by disciplined target-leakage auditing (city price levels in the MVP are computed from training rows only, passed at serve time; we never recompute using serving data).

---

## 8. Distributed data (Spark / Kafka) — not in MVP, but how I'd build it

Since this is an in-process MVP, I skipped Spark + Kafka — but they sit in a real architecture like this:

- **Ingest.** MLS feeds + broker events land in Kafka topics (`listing.events.v1`, `user.activity.v1`). Schema-registered with Avro/Protobuf so producers can't break consumers silently.
- **Offline / batch features.** Spark jobs (triggered by Airflow) read from the data lake (Iceberg on S3), compute training-time features, write to the offline feature store. The AVM retrain DAG reads from there.
- **Online / real-time features.** A Kafka Streams (or Flink) topology materializes the same features into Redis / DynamoDB on a seconds-latency path. Training parity is enforced by running the *same feature-transform library* in both worlds — ideally compiled once (e.g. via a shared Python package or a DSL compiled to both Spark + Flink).
- **Backfill.** New feature = backfill the offline store, re-train, run shadow deploy, promote. This is the #1 reason teams wish they'd invested in Spark + feature stores earlier.

SQL proficiency shows up at all layers: ad-hoc analytics, offline feature definition (usually as dbt models), and — critically — the offline eval gate queries that compare model versions.

---

## 9. A/B testing

The MVP implements hash-based deterministic bucketing salted by experiment ID. In production I'd layer a lightweight experimentation platform on top (or integrate with whatever Redfin already uses) to handle:

1. **Mutually exclusive experiment slots** so concurrent experiments don't interfere.
2. **Auto-sizing.** Pre-compute required sample size per variant for the target MDE.
3. **CUPED-style variance reduction** for metrics like "session-level conversion."
4. **SRM (sample ratio mismatch) checks.** If the split drifts from 90/10 → 88/12, something's wrong with the router.

For the interview, the MVP shows the kernel: the router is deterministic, the `/v1/ab/config` endpoint is how we'd expose experiment state, and `experiment_id` gives us independent bucketing across experiments.

---

## 10. Roadmap (what I'd ship in my first 90 days)

**Week 1-2: listen + orient.** Read existing AVM+recommender codebases, shadow on-call, understand where the team is losing time. Probably includes: auditing model registry hygiene, training data lineage, and the most recent production incident timelines.

**Weeks 3-6: first real wins.**
- **Cut AVM p95 latency by 30%.** Usually possible through batch-predict endpoint tightening, feature-store caching, ONNX-export of GBDT where relevant.
- **Add drift monitoring** to the top 10 features if it's not already there — biggest marginal-cost ROI in MLOps.
- **Wire up a shadow-deploy path** for the AVM (if not already present). This is what unlocks faster iteration — candidates can be tested against real traffic without customer risk.

**Weeks 7-12: invest in the platform.**
- **Shared feature transform library** across training + serving (kill skew by construction).
- **Automated rollback** tied to SLO burn rates.
- **Experimentation velocity.** Make shipping a new AVM candidate a 1-day turnaround, not a 2-week one.

**Quarter 2:**
- Cross-pollinate between AVM and recommender: the user-taste embeddings the recommender learns are *features* that probably help the AVM (price elasticity by preference cluster).
- Start exploring multi-modal: image embeddings for "home condition" as an AVM feature. CNN ensembles are well-worn here — the integration work is the hard part.

---

## 11. How to run the MVP

```bash
cd redfin-ml-mvp
pip install -r requirements.txt
python -m data.generate            # → data/homes.parquet (50k rows)
python -m src.avm.train            # registers AVM v1, promotes to production
python -m src.recommender.train    # registers recommender v1, promotes to production
uvicorn src.serving.app:app --reload
# Open ui/index.html (API URL defaults to localhost:8000)
```

Or the stack with monitoring:
```bash
docker-compose up --build
# API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

Tests:
```bash
pytest                              # full suite
pytest tests/test_drift.py -v       # just the drift detector
```

---

## 12. Things I'd do differently with more time

- **Real feature store.** Feast or Tecton instead of the in-process stub.
- **MLflow Model Registry** instead of the filesystem registry. The MVP's registry API deliberately mirrors MLflow's so the swap is surface-level.
- **ONNX / TreeLite export** for the LightGBM model to cut cold-start + serve latency.
- **A CUPED-style experimentation layer** wrapping the A/B router.
- **Real feature drift on categoricals** — the MVP's PSI treats categoricals as ordinal via the code mapping; for production we'd add chi-square testing on the raw categoricals too.
- **Cost tracking.** Per-prediction cost (CPU-seconds) is a metric worth exposing; AVM inference is a line item on a finance team's ledger at Redfin's scale.
