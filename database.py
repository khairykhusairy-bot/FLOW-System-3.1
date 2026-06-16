"""
FLOW — Flood Level Observation Warning System
Database Module: SQLite logging for monitoring data
"""

import sqlite3
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "flow_monitoring.db"


def _get_model_class_columns() -> List[str]:
    """
    Read class names from best.pt and return safe SQL column names.
    Each class becomes a column like 'cls_bottle', 'cls_tire', etc.
    The 'cls_' prefix avoids collisions with existing fixed columns.
    Returns an empty list if the model cannot be read.
    """
    try:
        from utils import get_model_class_names
        names = get_model_class_names("best.pt")
        # Sanitise: lowercase, spaces→underscore, only alphanumeric+underscore
        import re
        safe = []
        for n in names:
            col = "cls_" + re.sub(r"[^a-z0-9_]", "_", n.lower().replace(" ", "_"))
            safe.append(col)
        return safe
    except Exception as e:
        print(f"[FLOW DB] Could not read model classes: {e}")
        return []


def init_db():
    """Initialize the SQLite database and create tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_roi_objects INTEGER DEFAULT 0,
            blockage_percentage REAL DEFAULT 0.0,
            rain_intensity TEXT DEFAULT 'No Rain',
            rain_intensity_value REAL DEFAULT 0.0,
            humidity REAL DEFAULT NULL,
            wind_speed REAL DEFAULT NULL,
            temperature REAL DEFAULT NULL,
            feels_like REAL DEFAULT NULL,
            flood_risk TEXT DEFAULT 'Low Risk',
            confidence REAL DEFAULT 0.0,
            alert_triggered INTEGER DEFAULT 0,
            alert_message TEXT DEFAULT '',
            location TEXT DEFAULT ''
        )
    """)

    # ── Migration: add fixed columns to existing databases ────────────────────
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(monitoring_logs)")}
    migrations = {
        "humidity":             "ALTER TABLE monitoring_logs ADD COLUMN humidity             REAL DEFAULT NULL",
        "wind_speed":           "ALTER TABLE monitoring_logs ADD COLUMN wind_speed           REAL DEFAULT NULL",
        "temperature":          "ALTER TABLE monitoring_logs ADD COLUMN temperature          REAL DEFAULT NULL",
        "feels_like":           "ALTER TABLE monitoring_logs ADD COLUMN feels_like           REAL DEFAULT NULL",
        "rain_intensity_value": "ALTER TABLE monitoring_logs ADD COLUMN rain_intensity_value REAL DEFAULT 0.0",
        "water_level_cm":       "ALTER TABLE monitoring_logs ADD COLUMN water_level_cm      REAL DEFAULT NULL",
        "water_level_trend":    "ALTER TABLE monitoring_logs ADD COLUMN water_level_trend   TEXT DEFAULT NULL",
        "water_level_status":   "ALTER TABLE monitoring_logs ADD COLUMN water_level_status  TEXT DEFAULT NULL",
        "water_rise_rate":      "ALTER TABLE monitoring_logs ADD COLUMN water_rise_rate     REAL DEFAULT NULL",
        "location":             "ALTER TABLE monitoring_logs ADD COLUMN location            TEXT DEFAULT ''",
        # Legacy columns kept for backward-compatibility with existing databases
        "bottles":              "ALTER TABLE monitoring_logs ADD COLUMN bottles             INTEGER DEFAULT 0",
        "plastic_waste":        "ALTER TABLE monitoring_logs ADD COLUMN plastic_waste        INTEGER DEFAULT 0",
        "logs_count":           "ALTER TABLE monitoring_logs ADD COLUMN logs_count           INTEGER DEFAULT 0",
        "branches":             "ALTER TABLE monitoring_logs ADD COLUMN branches             INTEGER DEFAULT 0",
        "trash":                "ALTER TABLE monitoring_logs ADD COLUMN trash                INTEGER DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing_cols:
            cursor.execute(sql)

    # ── Dynamic migration: add one column per class in best.pt ────────────────
    # Any new class added to best.pt automatically gets its own column here.
    for col in _get_model_class_columns():
        if col not in existing_cols:
            cursor.execute(
                f"ALTER TABLE monitoring_logs ADD COLUMN {col} INTEGER DEFAULT 0"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_monitoring_data(data: Dict):
    """Insert a monitoring snapshot into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── Fixed columns always written ──────────────────────────────────────────
    fixed_cols = [
        "timestamp", "total_roi_objects", "blockage_percentage",
        "rain_intensity", "rain_intensity_value",
        "humidity", "wind_speed", "temperature", "feels_like",
        "flood_risk", "confidence", "alert_triggered", "alert_message",
        "water_level_cm", "water_level_trend", "water_level_status",
        "water_rise_rate", "location",
    ]
    fixed_values = [
        data.get("timestamp", datetime.now().isoformat()),
        data.get("total_roi_objects", 0),
        data.get("blockage_percentage", 0.0),
        data.get("rain_intensity", "No Rain"),
        data.get("rain_intensity_value", 0.0),
        data.get("humidity", None),
        data.get("wind_speed", None),
        data.get("temperature", None),
        data.get("feels_like", None),
        data.get("flood_risk", "Low Risk"),
        data.get("confidence", 0.0),
        int(data.get("alert_triggered", False)),
        data.get("alert_message", ""),
        data.get("water_level_cm", None),
        data.get("water_level_trend", None),
        data.get("water_level_status", None),
        data.get("water_rise_rate", None),
        data.get("location", ""),
    ]

    # ── Dynamic per-class columns from best.pt ────────────────────────────────
    # roi_counts is a dict {class_name: count} built live from the model output.
    # Each class gets its own 'cls_<name>' column; unknown classes are skipped
    # gracefully (the column may not exist yet until init_db runs again).
    import re
    roi_counts: Dict = data.get("roi_counts", {})
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(monitoring_logs)")}

    dynamic_cols = []
    dynamic_values = []
    for class_name, count in roi_counts.items():
        col = "cls_" + re.sub(r"[^a-z0-9_]", "_", class_name.lower().replace(" ", "_"))
        if col in existing_cols:
            dynamic_cols.append(col)
            dynamic_values.append(int(count))

    all_cols   = fixed_cols + dynamic_cols
    all_values = fixed_values + dynamic_values
    placeholders = ", ".join(["?"] * len(all_cols))
    col_str = ", ".join(all_cols)

    cursor.execute(
        f"INSERT INTO monitoring_logs ({col_str}) VALUES ({placeholders})",
        all_values,
    )
    conn.commit()
    conn.close()


def log_alert(alert_type: str, message: str, severity: str):
    """Log an alert event."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alert_history (timestamp, alert_type, message, severity)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), alert_type, message, severity))
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 50) -> List[Dict]:
    """Retrieve recent monitoring logs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM monitoring_logs
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_alerts(limit: int = 20) -> List[Dict]:
    """Retrieve recent alerts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM alert_history
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def export_to_csv(filepath: str = "flow_export.csv"):
    """Export all monitoring logs to CSV."""
    logs = get_recent_logs(limit=10000)
    if not logs:
        return False
    fieldnames = list(logs[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)
    return True


def get_stats_summary() -> Dict:
    """Get aggregate statistics from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total_records,
            AVG(blockage_percentage) as avg_blockage,
            MAX(blockage_percentage) as max_blockage,
            AVG(rain_intensity_value) as avg_rain,
            SUM(alert_triggered) as total_alerts
        FROM monitoring_logs
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "total_records": row[0],
            "avg_blockage": round(row[1] or 0, 2),
            "max_blockage": round(row[2] or 0, 2),
            "avg_rain": round(row[3] or 0, 3),
            "total_alerts": row[4] or 0
        }
    return {}
