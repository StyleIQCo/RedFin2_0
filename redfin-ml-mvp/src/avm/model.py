"""The AVM model object.

Wraps a LightGBM regressor + two "quantile twins" for a confidence interval.
This is the shape a real AVM takes in prod: a point estimate plus a 5/95 band
so the UI can show customers "We estimate $X, with 90% confidence it's between
$A and $B." Without that band, the product team can't reason about risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.avm.features import AVM_FEATURES, build_features


@dataclass
class AVMPrediction:
    point: float
    lower: float
    upper: float
    model_name: str
    model_version: int
    feature_contributions: dict[str, float]


class AVMModel:
    """LightGBM regression on log-price + quantile twins for CIs."""

    def __init__(
        self,
        booster: lgb.Booster,
        lower_booster: lgb.Booster,
        upper_booster: lgb.Booster,
        city_price_level: dict[str, float],
        feature_names: list[str] | None = None,
    ) -> None:
        self.booster = booster
        self.lower_booster = lower_booster
        self.upper_booster = upper_booster
        self.city_price_level = city_price_level
        self.feature_names = feature_names or AVM_FEATURES

    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> "AVMModel":
        from src.avm.features import compute_city_price_level

        city_map = compute_city_price_level(df)
        X = build_features(df, city_price_level=city_map)
        y = np.log(df["price"].to_numpy())

        default_params = {
            "objective": "regression",
            "metric": "mape",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        dset = lgb.Dataset(X, y, feature_name=AVM_FEATURES)
        main_booster = lgb.train(default_params, dset, num_boost_round=250)

        def _q(alpha: float) -> lgb.Booster:
            q_params = {**default_params, "objective": "quantile", "alpha": alpha, "metric": "quantile"}
            return lgb.train(q_params, dset, num_boost_round=200)

        lower = _q(0.05)
        upper = _q(0.95)
        return cls(main_booster, lower, upper, city_map)

    # --- prediction API ---
    def _prepare(self, record_or_df: dict | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([record_or_df]) if isinstance(record_or_df, dict) else record_or_df
        return build_features(df, city_price_level=self.city_price_level)

    def predict(self, record_or_df: dict | pd.DataFrame, model_name: str = "avm-lgbm", model_version: int = 0) -> AVMPrediction:
        X = self._prepare(record_or_df)
        point = float(np.exp(self.booster.predict(X)[0]))
        lower = float(np.exp(self.lower_booster.predict(X)[0]))
        upper = float(np.exp(self.upper_booster.predict(X)[0]))
        # Feature attributions via SHAP-like leaf path — cheaply approximated via gain contribution.
        contributions = self._feature_contributions(X)
        return AVMPrediction(
            point=point,
            lower=lower,
            upper=upper,
            model_name=model_name,
            model_version=model_version,
            feature_contributions=contributions,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        return pd.DataFrame({
            "point": np.exp(self.booster.predict(X)),
            "lower": np.exp(self.lower_booster.predict(X)),
            "upper": np.exp(self.upper_booster.predict(X)),
        })

    def _feature_contributions(self, X: pd.DataFrame) -> dict[str, float]:
        """Per-prediction contribution from LightGBM's `pred_contrib=True`."""
        contrib = self.booster.predict(X, pred_contrib=True)[0]  # last value is bias
        return {name: float(val) for name, val in zip(self.feature_names, contrib[:-1])}

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Offline eval — MAPE, median APE, coverage of the 90% CI."""
        preds = self.predict_batch(df)
        y = df["price"].to_numpy()
        ape = np.abs(preds["point"].to_numpy() - y) / y
        covered = ((y >= preds["lower"]) & (y <= preds["upper"])).mean()
        return {
            "mape": float(ape.mean()),
            "median_ape": float(np.median(ape)),
            "p90_coverage": float(covered),
            "n": int(len(df)),
        }
