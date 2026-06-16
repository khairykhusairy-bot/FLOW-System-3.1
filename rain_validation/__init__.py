"""
FLOW — Flood Level Observation Warning System
rain_validation package

Exposes the three validators and the composite orchestrator.
"""

from rain_validation.visibility          import VisibilityValidator
from rain_validation.surface_disturbance import SurfaceDisturbanceValidator
from rain_validation.rain_streaks        import RainStreakDetector
from rain_validation.composite           import (
    CompositeRainValidator,
    render_cv_validation_panel,
    RISK_LOW, RISK_MODERATE, RISK_HIGH, RISK_CRITICAL,
)

__all__ = [
    "VisibilityValidator",
    "SurfaceDisturbanceValidator",
    "RainStreakDetector",
    "CompositeRainValidator",
    "render_cv_validation_panel",
    "RISK_LOW", "RISK_MODERATE", "RISK_HIGH", "RISK_CRITICAL",
]
