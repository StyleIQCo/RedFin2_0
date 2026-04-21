"""Prometheus metrics singleton.

Exposes:
  - request counter  (by endpoint, model, version, variant, status)
  - request latency histogram
  - prediction histogram (AVM point estimate — lets us watch for distribution drift in outputs too)
  - drift gauge (max PSI across features)
  - model load counter
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class MetricsRegistry:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.request_count = Counter(
            "redfin_ml_requests_total",
            "Requests to the ML serving API.",
            ["endpoint", "model", "version", "variant", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "redfin_ml_request_latency_seconds",
            "End-to-end request latency (seconds).",
            ["endpoint", "model", "variant"],
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry,
        )
        self.avm_prediction = Histogram(
            "redfin_ml_avm_prediction_usd",
            "AVM point predictions (USD). Watches for output drift / runaway predictions.",
            buckets=(
                100_000, 250_000, 500_000, 750_000, 1_000_000,
                1_500_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000,
            ),
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "redfin_ml_drift_max_psi",
            "Maximum PSI across monitored features.",
            ["model"],
            registry=self.registry,
        )
        self.model_loaded = Counter(
            "redfin_ml_model_loads_total",
            "Count of model-load events (useful for detecting unexpected reloads).",
            ["model", "version", "stage"],
            registry=self.registry,
        )


metrics_registry = MetricsRegistry()
