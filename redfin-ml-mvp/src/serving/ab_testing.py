"""A/B testing router.

Simple deterministic hash-based bucketing:

    variant = "challenger" if hash(request_id) % 100 < 100*split else "champion"

Deterministic bucketing is non-negotiable: the same user/session must get the
same variant within an experiment window, or everything downstream breaks
(experiment stats, session-level metrics, user-level A/B cohort reports).

We also support **shadow mode**: the challenger predicts alongside the champion
but its response isn't returned to the user. Use this for the first week
after promoting a new model — no customer risk, but we can compare latency
and prediction distributions against production traffic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

Variant = Literal["champion", "challenger"]


@dataclass
class ABDecision:
    variant: Variant
    shadow: bool  # If True, challenger was run but champion's response is returned


class ABRouter:
    def __init__(
        self,
        traffic_split: float = 0.10,
        shadow: bool = False,
        experiment_id: str = "default",
    ) -> None:
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be in [0, 1]")
        self.traffic_split = traffic_split
        self.shadow = shadow
        self.experiment_id = experiment_id

    def _bucket(self, key: str) -> int:
        # Salt with experiment_id so different experiments are independent.
        raw = f"{self.experiment_id}:{key}".encode()
        h = hashlib.sha256(raw).hexdigest()
        return int(h[:8], 16) % 10_000  # finer granularity than mod 100

    def route(self, key: str) -> ABDecision:
        if self.traffic_split == 0.0:
            return ABDecision(variant="champion", shadow=self.shadow)
        in_challenger = self._bucket(key) < int(self.traffic_split * 10_000)
        if in_challenger:
            return ABDecision(variant="challenger", shadow=self.shadow)
        return ABDecision(variant="champion", shadow=False)

    def describe(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "traffic_split": self.traffic_split,
            "shadow": self.shadow,
        }
