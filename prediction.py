"""
FLOW — Flood Level Observation Warning System
Prediction Module: Rule-based flood risk predictor

Integrates with FloodRiskEngine (flood_risk_engine.py) to fuse:
  • Rule-based combined risk score (transparent, no circular ML)
  • Layer 3 Integrated Flood Probability:
      P = 0.6 × RainfallRisk + 0.2 × WaterLevelRisk + 0.2 × BlockageRisk

Fusion weights:  0.35 × rule_score  +  0.65 × engine_probability
The engine (deterministic, physically calibrated) carries the majority weight.
The rule score provides a fast per-frame debris/blockage signal.

The fused result is only active when monitoring is started (START clicked).
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


# ─── Risk Labels ───────────────────────────────────────────────────────────────
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {
    "Low Risk":    "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk":   "#e74c3c",
}

# ─── Combined score weights (must sum to 1.0) ──────────────────────────────────
# Blockage is the strongest local signal the camera can directly observe.
_W_BLOCKAGE  = 0.35
_W_RAIN      = 0.30
_W_WATER     = 0.20
_W_ROI_COUNT = 0.15

# Classification thresholds on the combined 0-1 score
_THRESH_LOW_MAX   = 0.30   # combined < 0.30 and secondary conditions → Low
_THRESH_HIGH_MIN  = 0.65   # combined > 0.65 or critical conditions   → High

# Secondary conditions for Low / High classification
_LOW_MAX_BLOCKAGE  = 30.0  # %
_LOW_MAX_RAIN      = 0.40  # normalised intensity
_HIGH_MIN_BLOCKAGE = 70.0  # %
_HIGH_RAIN_AND_BLOCK_RAIN    = 0.75
_HIGH_RAIN_AND_BLOCK_BLOCKAGE = 45.0


class FloodPredictor:
    """
    Rule-based flood risk predictor with temporal smoothing.

    The combined score is a normalised, dimensionless weighted sum of four
    observable features.  Classification thresholds are applied directly —
    no sklearn dependency, no synthetic training loop.

    Public API is identical to the previous ML-based version so main.py
    requires no changes.
    """

    def __init__(self):
        self.is_ready = True
        self._last_prediction = "Low Risk"
        self._last_confidence = 0.92
        self._smoothing_buffer: deque = deque(maxlen=5)

    # ── Feature builder ────────────────────────────────────────────────────────

    def _build_features(
        self,
        roi_count: int,
        blockage_pct: float,
        rain_intensity: float,
        water_level: float,
    ) -> Dict:
        density  = blockage_pct / max(roi_count + 1, 1)
        combined = (
            _W_BLOCKAGE  * (blockage_pct / 100.0)
            + _W_RAIN      * rain_intensity
            + _W_WATER     * water_level
            + _W_ROI_COUNT * min(roi_count / 30.0, 1.0)
        )
        return {
            "roi_count":    roi_count,
            "blockage_pct": blockage_pct,
            "rain":         rain_intensity,
            "water":        water_level,
            "density":      round(density, 4),
            "combined":     round(min(combined, 1.0), 4),
        }

    # ── Classification ─────────────────────────────────────────────────────────

    @staticmethod
    def _classify(combined: float, blockage_pct: float, rain_intensity: float) -> int:
        """
        Return 0 (Low), 1 (Medium), or 2 (High).

        Rules are applied in strict priority order — High overrides Medium,
        Medium overrides the Low check.
        """
        # High Risk conditions
        if (
            combined > _THRESH_HIGH_MIN
            or blockage_pct > _HIGH_MIN_BLOCKAGE
            or (rain_intensity > _HIGH_RAIN_AND_BLOCK_RAIN and blockage_pct > _HIGH_RAIN_AND_BLOCK_BLOCKAGE)
        ):
            return 2

        # Low Risk conditions (all must hold)
        if combined < _THRESH_LOW_MAX and blockage_pct < _LOW_MAX_BLOCKAGE and rain_intensity < _LOW_MAX_RAIN:
            return 0

        return 1  # Medium Risk

    # ── Approximate per-class probabilities ────────────────────────────────────

    @staticmethod
    def _score_to_proba(combined: float, pred_class: int) -> List[float]:
        """
        Derive approximate probabilities from the combined score so callers
        that display confidence bars still get sensible values.
        """
        if pred_class == 0:    # Low
            p_low  = max(0.70, 1.0 - combined * 2.0)
            p_high = min(0.05, combined * 0.3)
            p_mid  = 1.0 - p_low - p_high
        elif pred_class == 2:  # High
            p_high = max(0.70, combined)
            p_low  = min(0.05, 1.0 - combined)
            p_mid  = 1.0 - p_high - p_low
        else:                  # Medium
            p_mid  = max(0.50, 1.0 - abs(combined - 0.475) * 2.0)
            p_low  = (1.0 - p_mid) * (1.0 - combined)
            p_high = (1.0 - p_mid) * combined

        return [round(max(0.0, p_low), 4),
                round(max(0.0, p_mid), 4),
                round(max(0.0, p_high), 4)]

    # ── Public predict ─────────────────────────────────────────────────────────

    def predict(
        self,
        roi_count: int,
        blockage_pct: float,
        rain_intensity: float,
        water_level: float = 0.5,
    ) -> Dict:
        """
        Predict flood risk level.

        Returns:
            {
                "risk":          "Low Risk" | "Medium Risk" | "High Risk",
                "confidence":    float (0-1),
                "probabilities": {"Low Risk": p, "Medium Risk": p, "High Risk": p},
                "risk_score":    float (0-1),
                "color":         str,
            }
        """
        f = self._build_features(roi_count, blockage_pct, rain_intensity, water_level)
        pred_class = self._classify(f["combined"], f["blockage_pct"], f["rain"])

        # Temporal smoothing — 5-frame majority vote to suppress jitter
        self._smoothing_buffer.append(pred_class)
        smoothed_class = max(set(self._smoothing_buffer), key=list(self._smoothing_buffer).count)

        proba      = self._score_to_proba(f["combined"], smoothed_class)
        risk_label = RISK_LABELS[smoothed_class]
        confidence = proba[smoothed_class]

        self._last_prediction = risk_label
        self._last_confidence = confidence

        return {
            "risk":         risk_label,
            "confidence":   round(confidence, 4),
            "probabilities": {
                "Low Risk":    proba[0],
                "Medium Risk": proba[1],
                "High Risk":   proba[2],
            },
            "risk_score": f["combined"],
            "color":      RISK_COLORS[risk_label],
        }

    # ── Fused prediction ───────────────────────────────────────────────────────

    def predict_fused(
        self,
        roi_count: int,
        blockage_pct: float,
        rain_intensity: float,
        water_level: float = 0.5,
        risk_engine=None,
    ) -> Dict:
        """
        Fused flood risk prediction combining:
          • Rule-based combined risk score (this class, 35%)
          • Layer 3 Integrated Flood Probability from FloodRiskEngine (65%)

        The engine carries the majority weight because it is calibrated to
        physical rainfall thresholds.  The rule score contributes the fast
        per-frame debris/blockage signal.

        Fusion rule:
          final_probability = 0.35 × rule_score + 0.65 × engine_probability

        Returns the same dict shape as predict() with two additional keys:
            "engine_probability"  : float (0–1) from FloodRiskEngine
            "fused_probability"   : float (0–1) final blended value
        """
        ml_result = self.predict(roi_count, blockage_pct, rain_intensity, water_level)

        if risk_engine is None:
            ml_result["engine_probability"] = None
            ml_result["fused_probability"]  = ml_result["risk_score"]
            return ml_result

        try:
            layer3 = risk_engine.integrated_flood_probability(
                water_level_norm=water_level,
                blockage_pct=blockage_pct,
                camera_rain_norm=rain_intensity,
            )
            engine_prob = layer3["probability"]
        except Exception:
            engine_prob = ml_result["risk_score"]

        rule_score = ml_result["risk_score"]
        fused = round(min(1.0, 0.35 * rule_score + 0.65 * engine_prob), 4)

        if fused < 0.30:
            fused_label = "Low Risk"
        elif fused < 0.60:
            fused_label = "Medium Risk"
        else:
            fused_label = "High Risk"

        return {
            **ml_result,
            "risk":               fused_label,
            "color":              RISK_COLORS[fused_label],
            "risk_score":         fused,
            "engine_probability": round(engine_prob, 4),
            "fused_probability":  fused,
            "ml_risk":            ml_result["risk"],
            "ml_risk_score":      rule_score,
        }

    # ── Compatibility properties ───────────────────────────────────────────────

    @property
    def last_risk(self) -> str:
        return self._last_prediction

    @property
    def last_confidence(self) -> float:
        return self._last_confidence
