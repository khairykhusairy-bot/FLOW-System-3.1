"""
FLOW — Flood Level Observation Warning System
Alerts Module: Smart threshold-based alert system
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import time


# ─── Alert Severity ────────────────────────────────────────────────────────────
SEVERITY_INFO    = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL= "CRITICAL"

SEVERITY_COLORS = {
    SEVERITY_INFO:     "#3498db",
    SEVERITY_WARNING:  "#f39c12",
    SEVERITY_CRITICAL: "#e74c3c",
}

SEVERITY_ICONS = {
    SEVERITY_INFO:     "ℹ",
    SEVERITY_WARNING:  "⚠",
    SEVERITY_CRITICAL: "🚨",
}

# ─── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "blockage_warning":  50.0,
    "blockage_critical": 75.0,
    # rain_intensity is normalised 0-1 where 25 mm/h = 1.0 (Layer 1 "High" boundary).
    # rain_warning  = 0.60 → 15 mm/h = Layer 1 "Moderate" start.
    # rain_critical = 1.00 → 25+ mm/h = Layer 1 "High", consistent with flash-flood territory.
    "rain_warning":      0.60,
    "rain_critical":     1.00,
    "roi_count_warning": 10,
    "roi_count_critical":20,
    "risk_medium":       "Medium Risk",
    "risk_high":         "High Risk",
}


class Alert:
    def __init__(self, alert_id: str, alert_type: str, message: str, severity: str):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.color = SEVERITY_COLORS[severity]
        self.icon = SEVERITY_ICONS[severity]

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.strftime("%H:%M:%S"),
            "color": self.color,
            "icon": self.icon,
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """
    Manages real-time alerts for the FLOW monitoring system.
    Triggers alerts based on threshold violations and tracks alert history.
    """

    def __init__(self, cooldown_seconds: float = 15.0):
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self._cooldowns: Dict[str, float] = {}
        self.cooldown_seconds = cooldown_seconds
        self._callbacks: List[Callable] = []
        self._alert_counter = 0

    def add_callback(self, cb: Callable):
        """Register a callback that fires when a new alert is triggered."""
        self._callbacks.append(cb)

    def _on_cooldown(self, key: str) -> bool:
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) < self.cooldown_seconds

    def _trigger(self, alert_type: str, message: str, severity: str):
        key = f"{alert_type}_{severity}"
        if self._on_cooldown(key):
            return None
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALT-{self._alert_counter:04d}",
            alert_type=alert_type,
            message=message,
            severity=severity,
        )
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self._cooldowns[key] = time.time()

        # Keep active alerts list short
        if len(self.active_alerts) > 10:
            self.active_alerts = self.active_alerts[-10:]

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                pass

        return alert

    def evaluate(
        self,
        blockage_pct: float,
        rain_intensity: float,
        roi_count: int,
        flood_risk: str,
        custom_thresholds: Optional[Dict] = None,
    ) -> List[Alert]:
        """
        Evaluate monitoring metrics against thresholds and trigger alerts.
        Returns list of new alerts triggered in this call.

        Active alerts whose conditions have since dropped below threshold are
        automatically removed so has_critical() / has_warning() reflect the
        current state, not historical state.
        """
        th = {**THRESHOLDS, **(custom_thresholds or {})}
        new_alerts = []

        # Track which (alert_type, severity) pairs are CURRENTLY above threshold.
        # Any active alert NOT in this set gets pruned at the end of the method.
        currently_active: set = set()

        # ── Blockage alerts ────────────────────────────────────────────────────
        if blockage_pct >= th["blockage_critical"]:
            currently_active.add(("blockage", SEVERITY_CRITICAL))
            a = self._trigger(
                "blockage",
                f"CRITICAL: River path blockage at {blockage_pct:.1f}% — immediate action required",
                SEVERITY_CRITICAL,
            )
            if a: new_alerts.append(a)
        elif blockage_pct >= th["blockage_warning"]:
            currently_active.add(("blockage", SEVERITY_WARNING))
            a = self._trigger(
                "blockage",
                f"High blockage detected in river path ({blockage_pct:.1f}%)",
                SEVERITY_WARNING,
            )
            if a: new_alerts.append(a)

        # ── Rainfall alerts ────────────────────────────────────────────────────
        if rain_intensity >= th["rain_critical"]:
            currently_active.add(("rainfall", SEVERITY_CRITICAL))
            a = self._trigger(
                "rainfall",
                f"CRITICAL: Extreme rainfall intensity ({rain_intensity:.3f}) — flash flood risk",
                SEVERITY_CRITICAL,
            )
            if a: new_alerts.append(a)
        elif rain_intensity >= th["rain_warning"]:
            currently_active.add(("rainfall", SEVERITY_WARNING))
            a = self._trigger(
                "rainfall",
                f"Heavy rainfall warning — intensity {rain_intensity:.3f}",
                SEVERITY_WARNING,
            )
            if a: new_alerts.append(a)

        # ── ROI object count alerts ────────────────────────────────────────────
        if roi_count >= th["roi_count_critical"]:
            currently_active.add(("debris", SEVERITY_CRITICAL))
            a = self._trigger(
                "debris",
                f"CRITICAL: {roi_count} debris objects detected in monitoring zone",
                SEVERITY_CRITICAL,
            )
            if a: new_alerts.append(a)
        elif roi_count >= th["roi_count_warning"]:
            currently_active.add(("debris", SEVERITY_WARNING))
            a = self._trigger(
                "debris",
                f"High debris accumulation — {roi_count} objects in ROI",
                SEVERITY_WARNING,
            )
            if a: new_alerts.append(a)

        # ── Flood risk alerts ──────────────────────────────────────────────────
        if flood_risk == "High Risk":
            currently_active.add(("flood_risk", SEVERITY_CRITICAL))
            a = self._trigger(
                "flood_risk",
                "⚠ FLOOD RISK: HIGH — Evacuate vulnerable areas immediately",
                SEVERITY_CRITICAL,
            )
            if a: new_alerts.append(a)
        elif flood_risk == "Medium Risk":
            currently_active.add(("flood_risk", SEVERITY_WARNING))
            a = self._trigger(
                "flood_risk",
                "Flood Risk: MEDIUM — Monitor situation closely",
                SEVERITY_WARNING,
            )
            if a: new_alerts.append(a)

        # ── Prune resolved alerts ──────────────────────────────────────────────
        # Keep only active alerts whose (type, severity) is still above threshold.
        # This ensures has_critical() / has_warning() clear the moment conditions
        # return to normal, rather than persisting until the list is manually cleared.
        self.active_alerts = [
            a for a in self.active_alerts
            if (a.alert_type, a.severity) in currently_active
        ]

        return new_alerts

    def acknowledge_alert(self, alert_id: str):
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True

    def clear_acknowledged(self):
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]

    def clear_all(self):
        self.active_alerts = []

    def get_active_alerts(self) -> List[Dict]:
        return [a.to_dict() for a in reversed(self.active_alerts)]

    def get_history(self, limit: int = 50) -> List[Dict]:
        return [a.to_dict() for a in reversed(self.alert_history[-limit:])]

    def has_critical(self) -> bool:
        return any(a.severity == SEVERITY_CRITICAL for a in self.active_alerts)

    def has_warning(self) -> bool:
        return any(a.severity == SEVERITY_WARNING for a in self.active_alerts)

    def alert_count(self) -> int:
        return len(self.active_alerts)
