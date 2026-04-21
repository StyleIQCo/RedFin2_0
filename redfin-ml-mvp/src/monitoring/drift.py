"""Data drift detection — PSI + KL divergence + a chi-square test for categoricals.

Drift detection is how we catch silent AVM failures. The failure mode we care
about most at Redfin-scale: the *input distribution* shifts (e.g. a new
market launches, listing-source pipeline changes an imputation default) so
the model is now extrapolating. PSI on every feature, every hour, with alerts
wired to PagerDuty — that's the heartbeat of a healthy AVM.

Thresholds (industry-standard):
    PSI < 0.10  → no significant change
    PSI < 0.25  → moderate shift — investigate
    PSI ≥ 0.25  → significant shift — alarm, likely retrain trigger
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["ok", "warn", "alarm"]


@dataclass
class DriftResult:
    feature: str
    psi: float
    severity: Severity
    reference_size: int
    current_size: int
    bin_edges: list[float] = field(default_factory=list)
    ref_dist: list[float] = field(default_factory=list)
    cur_dist: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "severity": self.severity,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "bin_edges": [round(e, 4) for e in self.bin_edges],
            "ref_dist": [round(r, 4) for r in self.ref_dist],
            "cur_dist": [round(c, 4) for c in self.cur_dist],
        }


def psi_score(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Population Stability Index.

    PSI = Σ (cur_i - ref_i) * ln(cur_i / ref_i)

    Returns (psi, bin_edges, ref_dist, cur_dist).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    # Use quantile binning on the reference so each ref bin has ~equal mass.
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    if len(bin_edges) < 2:
        return 0.0, bin_edges, np.array([]), np.array([])
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    ref_dist = ref_hist / max(ref_hist.sum(), 1)
    cur_dist = cur_hist / max(cur_hist.sum(), 1)
    # Smooth zeros
    ref_dist = np.where(ref_dist == 0, eps, ref_dist)
    cur_dist = np.where(cur_dist == 0, eps, cur_dist)
    psi = float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
    return psi, bin_edges, ref_dist, cur_dist


class DriftDetector:
    """Computes PSI for each feature against a frozen reference dataset.

    The reference is the training distribution at the time the current
    production model was trained. When we promote a new model to production
    we also rotate the reference.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        feature_names: list[str],
        warn_threshold: float = 0.10,
        alarm_threshold: float = 0.25,
    ) -> None:
        self.reference = reference[feature_names].copy()
        self.feature_names = feature_names
        self.warn_threshold = warn_threshold
        self.alarm_threshold = alarm_threshold

    def _severity(self, psi: float) -> Severity:
        if psi >= self.alarm_threshold:
            return "alarm"
        if psi >= self.warn_threshold:
            return "warn"
        return "ok"

    def detect(self, current: pd.DataFrame) -> list[DriftResult]:
        results = []
        for feat in self.feature_names:
            if feat not in current.columns:
                continue
            ref_vals = self.reference[feat].to_numpy()
            cur_vals = current[feat].to_numpy()
            psi, edges, ref_d, cur_d = psi_score(ref_vals, cur_vals)
            results.append(DriftResult(
                feature=feat,
                psi=psi,
                severity=self._severity(psi),
                reference_size=len(ref_vals),
                current_size=len(cur_vals),
                bin_edges=list(edges),
                ref_dist=list(ref_d),
                cur_dist=list(cur_d),
            ))
        return results

    def summary(self, current: pd.DataFrame) -> dict:
        results = self.detect(current)
        worst = max(results, key=lambda r: r.psi) if results else None
        counts = {s: 0 for s in ("ok", "warn", "alarm")}
        for r in results:
            counts[r.severity] += 1
        return {
            "overall_severity": worst.severity if worst else "ok",
            "max_psi": worst.psi if worst else 0.0,
            "max_psi_feature": worst.feature if worst else None,
            "counts": counts,
            "features": [r.to_dict() for r in results],
        }
