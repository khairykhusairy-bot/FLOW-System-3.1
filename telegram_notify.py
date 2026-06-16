"""
FLOW — Flood Level Observation Warning System
Telegram Notification Module (Auto-Subscribe Edition)

How it works:
  1. Bot token is hardcoded — no manual input needed.
  2. A background thread polls /getUpdates every 5 seconds.
  3. Any user who sends /start to the bot is automatically added
     to the subscriber list and immediately receives a welcome message.
  4. When flood risk reaches High Risk, ALL subscribers get an alert.
  5. Periodic reminders are sent while High Risk persists.
  6. An all-clear is sent when risk drops back down.
"""

import requests
import time
import threading
import json
import os
from datetime import datetime
from typing import Optional, Dict, Set

# ─── Hardcoded Bot Token ───────────────────────────────────────────────────────
BOT_TOKEN = "8677038182:AAFXA-tY2UbBZe2Xm4NlidhQR2wxLaVS-aA"
BASE_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ─── Persistence ──────────────────────────────────────────────────────────────
SUBSCRIBERS_FILE    = "flow_subscribers.json"   # saved next to main.py
REMINDER_INTERVAL   = 300    # seconds between reminder messages (5 min)
POLL_INTERVAL       = 5      # seconds between /getUpdates polls


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_subscribers() -> Set[str]:
    """Load persisted subscriber chat IDs from disk."""
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                data = json.load(f)
            return set(str(c) for c in data.get("chat_ids", []))
        except Exception:
            pass
    return set()


def _save_subscribers(chat_ids: Set[str]):
    """Persist subscriber chat IDs to disk."""
    try:
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump({"chat_ids": list(chat_ids)}, f)
    except Exception as e:
        print(f"[FLOW-Telegram] Could not save subscribers: {e}")


def _post(endpoint: str, data: dict) -> bool:
    """POST to Telegram API. Returns True on success."""
    try:
        r = requests.post(f"{BASE_URL}/{endpoint}", data=data, timeout=10)
        return r.ok
    except Exception as e:
        print(f"[FLOW-Telegram] POST error ({endpoint}): {e}")
        return False


def _send_to(chat_id: str, text: str) -> bool:
    """Send an HTML message to a single chat ID."""
    return _post("sendMessage", {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    })


def _broadcast(text: str, subscribers: Set[str]):
    """Send a message to all subscribers in parallel threads."""
    for cid in list(subscribers):
        threading.Thread(
            target=_send_to, args=(cid, text), daemon=True
        ).start()


# ─── Message Builders ──────────────────────────────────────────────────────────

def _welcome_msg(location: str) -> str:
    return (
        "🌊 <b>Welcome to FLOW — Flood Monitoring System</b>\n\n"
        f"✅ You are now subscribed to flood alerts for <b>{location}</b>.\n\n"
        "You will automatically receive:\n"
        "  ⚠️ Watch notices when flood risk reaches <b>MEDIUM</b>\n"
        "  🚨 Alerts when flood risk reaches <b>HIGH</b>\n"
        "  🔁 Reminder messages every 5 minutes while risk is high\n"
        "  ✅ All-clear when risk drops back down\n\n"
        "No action needed — just keep this chat open.\n\n"
        "To stop alerts, send /stop\n"
        "To check system status, send /status\n\n"
        "<i>@Aiflowsystembot — FLOW Monitoring System</i>"
    )


def _stop_msg() -> str:
    return (
        "🔕 <b>Unsubscribed from FLOW Alerts</b>\n\n"
        "You will no longer receive flood notifications.\n"
        "Send /start anytime to re-subscribe.\n\n"
        "<i>FLOW Monitoring System</i>"
    )


def _medium_risk_msg(
    blockage_pct: float,
    rain_intensity: float,
    roi_count: int,
    water_level_cm: Optional[float],
    wl_trend: str,
    location: str = "Unknown Location",
) -> str:
    now      = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    rain_pct = rain_intensity * 100
    rain_icon = "⛈" if rain_intensity >= 0.85 else ("🌧" if rain_intensity >= 0.60 else "🌦")
    wl_str   = f"{water_level_cm:.1f} cm ({wl_trend})" if water_level_cm is not None else "N/A"

    return (
        f"⚠️ <b>FLOOD WATCH — MEDIUM RISK DETECTED</b>\n\n"
        f"📍 <b>Location:</b> {location}\n"
        f"🕐 <b>Time:</b> {now}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🌊 <b>Flood Risk:</b>  <code>Medium Risk</code>\n\n"
        f"📊 <b>Sensor Readings</b>\n"
        f"  🪵 River Blockage : <b>{blockage_pct:.1f}%</b>\n"
        f"  {rain_icon} Rain Intensity : <b>{rain_pct:.1f}%</b>\n"
        f"  📦 Debris in ROI  : <b>{roi_count}</b> objects\n"
        f"  💧 Water Level    : <b>{wl_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👁 <b>Monitor the situation closely.</b>\n\n"
        f"<i>FLOW Monitoring System</i>"
    )


def _alert_msg(
    risk: str,
    blockage_pct: float,
    rain_intensity: float,
    roi_count: int,
    confidence: float,
    water_level_cm: Optional[float],
    wl_trend: str,
    is_reminder: bool,
    location: str = "Unknown Location",
) -> str:
    now       = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    header    = "🔁 <b>FLOOD RISK REMINDER</b>" if is_reminder else "🚨 <b>FLOOD ALERT — HIGH RISK DETECTED</b>"
    rain_pct  = rain_intensity * 100
    rain_icon = "⛈" if rain_intensity >= 0.85 else ("🌧" if rain_intensity >= 0.60 else "🌦")
    wl_str    = f"{water_level_cm:.1f} cm ({wl_trend})" if water_level_cm is not None else "N/A"

    return (
        f"{header}\n\n"
        f"📍 <b>Location:</b> {location}\n"
        f"🕐 <b>Time:</b> {now}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🌊 <b>Flood Risk:</b>  <code>{risk}</code>\n"
        f"🎯 <b>Confidence:</b> {confidence*100:.1f}%\n\n"
        f"📊 <b>Sensor Readings</b>\n"
        f"  🪵 River Blockage : <b>{blockage_pct:.1f}%</b>\n"
        f"  {rain_icon} Rain Intensity : <b>{rain_pct:.1f}%</b>\n"
        f"  📦 Debris in ROI  : <b>{roi_count}</b> objects\n"
        f"  💧 Water Level    : <b>{wl_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⚠️ <b>Immediate action may be required.</b>\n\n"
        f"🆘 Emergency : <b>999</b>\n"
        f"🚒 Bomba     : <b>994</b>\n"
        f"🏛 NADMA     : <b>03-8064 2400</b>\n\n"
        f"<i>FLOW Monitoring System</i>"
    )


def _all_clear_msg(risk: str, location: str = "Unknown Location") -> str:
    return (
        f"✅ <b>FLOOD RISK REDUCED</b>\n\n"
        f"📍 {location}\n"
        f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M:%S')}\n\n"
        f"Flood risk has dropped to <b>{risk}</b>.\n"
        f"Continue monitoring the situation.\n\n"
        f"<i>FLOW Monitoring System</i>"
    )


# ─── Main Notifier Class ───────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Auto-subscribe Telegram notifier for FLOW.

    • Polls /getUpdates in a background thread.
    • Any user who sends /start is auto-added; /stop removes them.
    • Subscribers are saved to flow_subscribers.json so they survive restarts.
    • Broadcasts alert/reminder/all-clear to all subscribers automatically.
    """

    def __init__(self):
        self._lock              = threading.Lock()
        self._subscribers:Set[str] = _load_subscribers()
        self._offset            = 0            # Telegram update offset
        self._in_high_risk      = False
        self._in_medium_risk    = False
        self._high_risk_since   = 0.0
        self._last_reminder_at  = 0.0
        self._send_errors       = []
        self._location          = "Kangar, Perlis"  # updated via set_location()

        # Start polling thread immediately
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()
        print(f"[FLOW-Telegram] Auto-subscriber mode active. "
              f"{len(self._subscribers)} existing subscriber(s).")

    # ── Properties ─────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Always True — token is hardcoded."""
        return True

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    @property
    def subscribers(self):
        with self._lock:
            return set(self._subscribers)

    @property
    def last_errors(self) -> list:
        with self._lock:
            return list(self._send_errors[-5:])

    def configure(self, *args, **kwargs):
        """No-op — token is hardcoded. Kept for API compatibility."""
        pass

    def set_location(self, location_name: str):
        """Update the monitored location name shown in all Telegram messages."""
        with self._lock:
            self._location = location_name or "Unknown Location"

    def set_reminder_interval(self, seconds: float):
        """Kept for API compatibility — reminder interval is fixed at 5 min."""
        pass

    def reset_state(self):
        """Reset risk tracking (call when monitoring stops)."""
        with self._lock:
            self._in_high_risk     = False
            self._in_medium_risk   = False
            self._high_risk_since  = 0.0
            self._last_reminder_at = 0.0

    # ── Polling Loop ───────────────────────────────────────────────────────────

    def _poll_loop(self):
        """Background thread: continuously polls Telegram for new messages."""
        while True:
            try:
                self._fetch_updates()
            except Exception as e:
                print(f"[FLOW-Telegram] Poll error: {e}")
            time.sleep(POLL_INTERVAL)

    def _fetch_updates(self):
        """Fetch new updates from Telegram and handle /start and /stop."""
        try:
            r = requests.get(
                f"{BASE_URL}/getUpdates",
                params={"offset": self._offset, "timeout": 4},
                timeout=10,
            )
            if not r.ok:
                return
            data = r.json()
        except Exception:
            return

        for update in data.get("result", []):
            self._offset = update["update_id"] + 1
            msg = update.get("message") or update.get("edited_message")
            if not msg:
                continue

            chat_id = str(msg["chat"]["id"])
            text    = msg.get("text", "").strip().lower()
            name    = msg["chat"].get("first_name", "there")

            if text.startswith("/start"):
                with self._lock:
                    is_new = chat_id not in self._subscribers
                    self._subscribers.add(chat_id)
                    _save_subscribers(self._subscribers)
                    loc = self._location
                if is_new:
                    print(f"[FLOW-Telegram] New subscriber: {chat_id} ({name})")
                threading.Thread(
                    target=_send_to, args=(chat_id, _welcome_msg(loc)), daemon=True
                ).start()

            elif text.startswith("/stop"):
                with self._lock:
                    self._subscribers.discard(chat_id)
                    _save_subscribers(self._subscribers)
                print(f"[FLOW-Telegram] Unsubscribed: {chat_id} ({name})")
                threading.Thread(
                    target=_send_to, args=(chat_id, _stop_msg()), daemon=True
                ).start()

            elif text.startswith("/status"):
                # Anyone can query current status
                subs = self.subscriber_count
                with self._lock:
                    loc = self._location
                status_text = (
                    f"🌊 <b>FLOW System Status</b>\n\n"
                    f"📡 Monitoring: Active\n"
                    f"👥 Subscribers: {subs}\n"
                    f"📍 Location: {loc}\n\n"
                    f"<i>@Aiflowsystembot — FLOW Monitoring System</i>"
                )
                threading.Thread(
                    target=_send_to, args=(chat_id, status_text), daemon=True
                ).start()

    # ── Alert Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        flood_result: Dict,
        blockage_pct: float,
        rain_intensity: float,
        roi_count: int,
        water_level_cm: Optional[float] = None,
        wl_trend: str = "Stable",
    ) -> Optional[str]:
        """
        Call every detection cycle.
        Automatically broadcasts to all subscribers when appropriate.
        Returns "entry" | "reminder" | "all_clear" | "medium_entry" | None.
        """
        risk       = flood_result.get("risk", "Low Risk")
        confidence = flood_result.get("confidence", 0.0)
        now        = time.time()

        with self._lock:
            subs = set(self._subscribers)
            loc  = self._location

        if not subs:
            return None   # nobody subscribed yet

        # ── Entry: just became High Risk ──────────────────────────────────────
        if risk == "High Risk" and not self._in_high_risk:
            self._in_high_risk     = True
            self._in_medium_risk   = False   # escalating out of medium
            self._high_risk_since  = now
            self._last_reminder_at = now
            msg = _alert_msg(
                risk, blockage_pct, rain_intensity, roi_count,
                confidence, water_level_cm, wl_trend, is_reminder=False,
                location=loc,
            )
            threading.Thread(
                target=_broadcast, args=(msg, subs), daemon=True
            ).start()
            print(f"[FLOW-Telegram] High Risk alert sent to {len(subs)} subscriber(s).")
            return "entry"

        # ── Reminder: still High Risk after REMINDER_INTERVAL ─────────────────
        if risk == "High Risk" and self._in_high_risk:
            if now - self._last_reminder_at >= REMINDER_INTERVAL:
                self._last_reminder_at = now
                msg = _alert_msg(
                    risk, blockage_pct, rain_intensity, roi_count,
                    confidence, water_level_cm, wl_trend, is_reminder=True,
                    location=loc,
                )
                threading.Thread(
                    target=_broadcast, args=(msg, subs), daemon=True
                ).start()
                print(f"[FLOW-Telegram] Reminder sent to {len(subs)} subscriber(s).")
                return "reminder"

        # ── All-clear: risk dropped from High ─────────────────────────────────
        if risk != "High Risk" and self._in_high_risk:
            self._in_high_risk = False
            # If dropping to Medium, mark state so we don't immediately re-fire
            # the medium entry alert on the next cycle — the all-clear covers it.
            self._in_medium_risk = (risk == "Medium Risk")
            msg = _all_clear_msg(risk, location=loc)
            threading.Thread(
                target=_broadcast, args=(msg, subs), daemon=True
            ).start()
            print(f"[FLOW-Telegram] All-clear sent to {len(subs)} subscriber(s).")
            return "all_clear"

        # ── Medium Risk entry: just reached Medium from Low ───────────────────
        if risk == "Medium Risk" and not self._in_medium_risk:
            self._in_medium_risk = True
            msg = _medium_risk_msg(
                blockage_pct, rain_intensity, roi_count,
                water_level_cm, wl_trend, location=loc,
            )
            threading.Thread(
                target=_broadcast, args=(msg, subs), daemon=True
            ).start()
            print(f"[FLOW-Telegram] Medium Risk watch sent to {len(subs)} subscriber(s).")
            return "medium_entry"

        # ── Medium Risk cleared: dropped back to Low ──────────────────────────
        if risk != "Medium Risk" and self._in_medium_risk:
            self._in_medium_risk = False

        return None

    # ── Test ───────────────────────────────────────────────────────────────────

    def send_test(self) -> tuple:
        """
        Send a test message to ALL current subscribers.
        Returns (success, message_string).
        """
        with self._lock:
            subs = set(self._subscribers)
            loc  = self._location

        if not subs:
            return False, "No subscribers yet. Ask someone to send /start to your bot first."

        msg = (
            "🌊 <b>FLOW Test Message</b>\n\n"
            f"✅ System is online and monitoring.\n"
            f"👥 {len(subs)} active subscriber(s).\n"
            f"📍 Location: {loc}\n\n"
            "You will receive alerts when flood risk reaches HIGH.\n\n"
            "<i>FLOW Monitoring System</i>"
        )
        _broadcast(msg, subs)
        return True, f"Test sent to {len(subs)} subscriber(s)!"
