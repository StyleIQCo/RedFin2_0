"""Drift detection tests — no drift should read low PSI; shifted data should alarm."""
import numpy as np
import pandas as pd

from src.monitoring.drift import DriftDetector, psi_score


def test_identical_distributions_have_zero_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=5000)
    cur = rng.normal(0, 1, size=5000)
    psi, _, _, _ = psi_score(ref, cur, n_bins=10)
    assert psi < 0.05, f"PSI on identical dists should be tiny, got {psi}"


def test_shifted_distribution_alarms():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=5000)
    # Mean shift of +1 sigma — clear "moderate" drift
    cur = rng.normal(1.0, 1, size=5000)
    psi, _, _, _ = psi_score(ref, cur, n_bins=10)
    assert psi > 0.25, f"Shifted dist should exceed alarm, got {psi}"


def test_detector_produces_per_feature_results():
    rng = np.random.default_rng(1)
    ref_df = pd.DataFrame({
        "sqft": rng.normal(2000, 400, 5000),
        "beds": rng.integers(1, 5, 5000),
        "baths": rng.integers(1, 5, 5000),
    })
    cur_df = pd.DataFrame({
        "sqft": rng.normal(2500, 400, 1000),   # shifted
        "beds": rng.integers(1, 5, 1000),       # same
        "baths": rng.integers(1, 5, 1000),      # same
    })
    detector = DriftDetector(ref_df, feature_names=["sqft", "beds", "baths"])
    summary = detector.summary(cur_df)
    assert summary["max_psi_feature"] == "sqft"
    assert summary["overall_severity"] in ("warn", "alarm")
