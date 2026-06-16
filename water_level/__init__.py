"""
FLOW — Water Level Module
water_level/__init__.py

Exposes the public API for the water level estimation subsystem.
Import the WaterLevelMonitor facade class for the simplest integration:

    from water_level import WaterLevelMonitor

    monitor = WaterLevelMonitor()
    monitor.calibration.load()

    # In your frame loop:
    result = monitor.process(frame)
    frame  = monitor.draw(frame, result)
"""

from water_level.monitor import WaterLevelMonitor
from water_level.calibration import WaterLevelCalibration
from water_level.detector import WaterlineDetector
from water_level.smoothing import WaterLevelSmoother
from water_level.trend_analysis import WaterLevelTrend
from water_level.visualization import draw_water_level_overlay

__all__ = [
    "WaterLevelMonitor",
    "WaterLevelCalibration",
    "WaterlineDetector",
    "WaterLevelSmoother",
    "WaterLevelTrend",
    "draw_water_level_overlay",
]
