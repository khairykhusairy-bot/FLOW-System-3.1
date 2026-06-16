"""
FLOW — Flood Level Observation Warning System
Weather Module: Dual-provider real-time weather

Providers:
  • Google Weather API  — true hourly (1-hour intervals)
  • OpenWeatherMap      — 3-hour intervals, wider forecast cards

Usage:
    from weather import WeatherService
    ws = WeatherService(provider="google")          # or "openweathermap"
    data = ws.get_current()
    forecast = ws.get_forecast()
"""

import urllib.request
import urllib.parse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from config import WEATHER_LOCATIONS, GOOGLE_WEATHER_API_KEY, OWM_API_KEY, WEATHER_PROVIDER

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False


# ─── Custom Location Persistence ──────────────────────────────────────────────
_CUSTOM_LOC_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flow_custom_locations.json"
)


def _load_custom_locations() -> Dict[str, tuple]:
    try:
        if os.path.exists(_CUSTOM_LOC_FILE):
            with open(_CUSTOM_LOC_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {k: tuple(v) for k, v in raw.items() if len(v) == 2}
    except Exception:
        pass
    return {}


def _save_custom_locations(locs: Dict[str, tuple]) -> None:
    try:
        with open(_CUSTOM_LOC_FILE, "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in locs.items()}, f, indent=2)
    except Exception as exc:
        print(f"[FLOW Weather] Could not save custom locations: {exc}")


def _delete_custom_location(name: str) -> None:
    locs = _load_custom_locations()
    locs.pop(name, None)
    _save_custom_locations(locs)


def _reverse_geocode(lat: float, lon: float) -> str:
    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=14&accept-language=en"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FLOW-FloodMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        addr = data.get("address", {})
        parts: List[str] = []
        for key in ("suburb", "city_district", "quarter", "town", "city", "county", "state"):
            val = addr.get(key, "")
            if val and val not in parts:
                parts.append(val)
                if len(parts) == 2:
                    break
        return ", ".join(parts) if parts else data.get("display_name", "")[:60]
    except Exception:
        return ""


# ─── Google Weather condition type → FLOW label + icon ───────────────────────
_GOOGLE_CONDITION_MAP: Dict[str, Dict] = {
    "CLEAR":                              {"label": "Clear Sky",              "day": "☀️",  "night": "🌙"},
    "MOSTLY_CLEAR":                       {"label": "Mainly Clear",           "day": "🌤️", "night": "🌙"},
    "PARTLY_CLOUDY":                      {"label": "Partly Cloudy",          "day": "⛅",  "night": "🌤️"},
    "MOSTLY_CLOUDY":                      {"label": "Mostly Cloudy",          "day": "☁️",  "night": "☁️"},
    "CLOUDY":                             {"label": "Overcast",               "day": "☁️",  "night": "☁️"},
    "FOG":                                {"label": "Fog",                    "day": "🌫️", "night": "🌫️"},
    "LIGHT_FOG":                          {"label": "Light Fog",              "day": "🌫️", "night": "🌫️"},
    "DRIZZLE":                            {"label": "Light Drizzle",          "day": "🌦️", "night": "🌦️"},
    "LIGHT_RAIN_AND_WIND":                {"label": "Light Rain",             "day": "🌧️", "night": "🌧️"},
    "RAIN":                               {"label": "Slight Rain",            "day": "🌧️", "night": "🌧️"},
    "LIGHT_RAIN":                         {"label": "Slight Rain",            "day": "🌧️", "night": "🌧️"},
    "MODERATE_RAIN":                      {"label": "Moderate Rain",          "day": "🌧️", "night": "🌧️"},
    "HEAVY_RAIN":                         {"label": "Heavy Rain",             "day": "🌧️", "night": "🌧️"},
    "RAIN_AND_WIND":                      {"label": "Heavy Rain",             "day": "🌧️", "night": "🌧️"},
    "HEAVY_RAIN_AND_WIND":                {"label": "Heavy Rain",             "day": "🌧️", "night": "🌧️"},
    "SHOWERS":                            {"label": "Slight Rain",            "day": "🌦️", "night": "🌦️"},
    "HEAVY_SHOWERS":                      {"label": "Heavy Rain",             "day": "🌧️", "night": "🌧️"},
    "FREEZING_DRIZZLE_FREEZING_RAIN":     {"label": "Freezing Rain",          "day": "🌨️", "night": "🌨️"},
    "SNOW":                               {"label": "Moderate Snow",          "day": "❄️",  "night": "❄️"},
    "LIGHT_SNOW":                         {"label": "Slight Snow",            "day": "🌨️", "night": "🌨️"},
    "HEAVY_SNOW":                         {"label": "Heavy Snow",             "day": "❄️",  "night": "❄️"},
    "SNOW_AND_WIND":                      {"label": "Heavy Snow",             "day": "❄️",  "night": "❄️"},
    "BLIZZARD":                           {"label": "Blizzard",               "day": "❄️",  "night": "❄️"},
    "THUNDERSTORM":                       {"label": "Thunderstorm",           "day": "⛈️", "night": "⛈️"},
    "THUNDERSTORM_AND_RAIN":              {"label": "Thunderstorm",           "day": "⛈️", "night": "⛈️"},
    "HEAVY_THUNDERSTORM_AND_RAIN":        {"label": "Heavy Thunderstorm",     "day": "⛈️", "night": "⛈️"},
    "LIGHT_THUNDERSTORM":                 {"label": "Light Thunderstorm",     "day": "⛈️", "night": "⛈️"},
    "LIGHT_THUNDERSTORM_RAIN":            {"label": "Light Thunderstorm Rain","day": "⛈️", "night": "⛈️"},
    "THUNDERSTORM_RAIN":                  {"label": "Thunderstorm Rain",      "day": "⛈️", "night": "⛈️"},
    "SCATTERED_THUNDERSTORMS":            {"label": "Scattered Thunderstorms","day": "⛈️", "night": "⛈️"},
    "ISOLATED_THUNDERSTORMS":             {"label": "Isolated Thunderstorms", "day": "⛈️", "night": "⛈️"},
    "THUNDERSHOWERS":                     {"label": "Thundershowers",         "day": "⛈️", "night": "⛈️"},
    "HEAVY_THUNDERSHOWERS":               {"label": "Heavy Thundershowers",   "day": "⛈️", "night": "⛈️"},
}


def _smart_condition_fallback(condition_type: str, is_day: bool) -> Dict:
    t = condition_type.upper()
    if "THUNDER" in t or "LIGHTNING" in t or "STORM" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "⛈️"}
    if "BLIZZARD" in t or "SNOW" in t or "SLEET" in t or "HAIL" in t or "ICE" in t or "FREEZ" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "❄️"}
    if "SHOWER" in t or "HEAVY_RAIN" in t or "DOWNPOUR" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "🌧️"}
    if "RAIN" in t or "DRIZZLE" in t or "PRECIP" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "🌧️"}
    if "FOG" in t or "MIST" in t or "HAZE" in t or "SMOKE" in t or "DUST" in t or "SAND" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "🌫️"}
    if "CLOUD" in t or "OVERCAST" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "☁️"}
    if "WIND" in t or "GUST" in t or "BREEZY" in t or "SQUALL" in t or "TORNADO" in t or "HURRICANE" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "💨"}
    if "CLEAR" in t or "SUNNY" in t or "FAIR" in t:
        return {"label": condition_type.replace("_", " ").title(), "icon": "☀️" if is_day else "🌙"}
    return {"label": condition_type.replace("_", " ").title(), "icon": "🌤️" if is_day else "🌙"}


def _google_condition(condition_type: str, is_day: bool) -> Dict:
    entry = _GOOGLE_CONDITION_MAP.get(condition_type.upper())
    if entry:
        return {"label": entry["label"], "icon": entry["day"] if is_day else entry["night"]}
    return _smart_condition_fallback(condition_type, is_day)


def rain_intensity_to_category(intensity: float) -> str:
    if intensity <= 0.0:   return "No Rain"
    elif intensity < 0.2:  return "Light Drizzle"
    elif intensity < 0.4:  return "Slight Rain"
    elif intensity < 0.6:  return "Moderate Rain"
    elif intensity < 0.8:  return "Heavy Rain"
    else:                  return "Violent Showers"


def _rain_to_intensity(mm_per_hour: float) -> float:
    return round(min(1.0, mm_per_hour / 25.0), 4)


# ─── OpenWeatherMap backend ────────────────────────────────────────────────────
class _OWMBackend:
    """Fetches weather from OpenWeatherMap (3-hour forecast intervals)."""

    CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

    _OWM_ICON_MAP = {
        200: ("Thunderstorm w/ Rain",  "⛈️"), 201: ("Thunderstorm w/ Rain",  "⛈️"),
        202: ("Heavy Thunderstorm",    "⛈️"), 210: ("Light Thunderstorm",    "⛈️"),
        211: ("Thunderstorm",          "⛈️"), 212: ("Heavy Thunderstorm",    "⛈️"),
        221: ("Ragged Thunderstorm",   "⛈️"), 230: ("Thunderstorm Drizzle",  "⛈️"),
        231: ("Thunderstorm Drizzle",  "⛈️"), 232: ("Heavy T-Storm Drizzle", "⛈️"),
        300: ("Light Drizzle",         "🌦️"), 301: ("Drizzle",               "🌦️"),
        302: ("Heavy Drizzle",         "🌧️"), 310: ("Light Drizzle Rain",    "🌧️"),
        311: ("Drizzle Rain",          "🌧️"), 312: ("Heavy Drizzle Rain",    "🌧️"),
        313: ("Shower Drizzle",        "🌧️"), 314: ("Heavy Shower Drizzle",  "🌧️"),
        321: ("Shower Drizzle",        "🌧️"), 500: ("Slight Rain",           "🌧️"),
        501: ("Moderate Rain",         "🌧️"), 502: ("Heavy Rain",            "🌧️"),
        503: ("Very Heavy Rain",       "🌧️"), 504: ("Extreme Rain",          "🌧️"),
        511: ("Freezing Rain",         "🌨️"), 520: ("Light Showers",         "🌦️"),
        521: ("Rain Showers",          "🌧️"), 522: ("Heavy Showers",         "🌧️"),
        531: ("Ragged Showers",        "🌧️"), 600: ("Slight Snow",           "❄️"),
        601: ("Moderate Snow",         "❄️"),  602: ("Heavy Snow",            "❄️"),
        611: ("Sleet",                 "🌨️"), 612: ("Light Sleet",           "🌨️"),
        613: ("Sleet Showers",         "🌨️"), 615: ("Light Rain & Snow",     "🌨️"),
        616: ("Rain & Snow",           "🌨️"), 620: ("Light Snow Showers",    "🌨️"),
        621: ("Snow Showers",          "❄️"),  622: ("Heavy Snow Showers",    "❄️"),
        701: ("Mist",                  "🌫️"), 711: ("Smoke",                 "🌫️"),
        721: ("Haze",                  "🌫️"), 731: ("Dust/Sand",             "🌫️"),
        741: ("Fog",                   "🌫️"), 751: ("Sand",                  "🌫️"),
        761: ("Dust",                  "🌫️"), 762: ("Volcanic Ash",          "🌫️"),
        771: ("Squalls",               "💨"),  781: ("Tornado",               "🌪️"),
        800: ("Clear Sky",             "☀️"),  801: ("Few Clouds (11-25%)",   "🌤️"),
        802: ("Scattered Clouds",      "⛅"),  803: ("Broken Clouds",         "☁️"),
        804: ("Overcast",              "☁️"),
    }

    def __init__(self, lat, lon, name, api_key, cache_ttl):
        self.lat = lat; self.lon = lon; self.name = name
        self.api_key = api_key; self.cache_ttl = cache_ttl
        self._current_cache:  Optional[Dict]       = None
        self._forecast_cache: Optional[List[Dict]] = None
        self._last_fetch:     float                = 0.0
        self._fetch_error:    Optional[str]        = None

    def maybe_refresh(self):
        if (time.time() - self._last_fetch) < self.cache_ttl:
            return
        try:
            self._fetch(); self._fetch_error = None
        except Exception as exc:
            self._fetch_error = str(exc)
            print(f"[FLOW Weather OWM] Fetch failed: {exc}")

    def _fetch_json(self, url: str) -> Dict:
        req = urllib.request.Request(url, headers={"User-Agent": "FLOW-FloodMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _owm_condition(self, weather_list: list, is_day: bool):
        if not weather_list:
            return "Clear Sky", "☀️" if is_day else "🌙"
        cid   = weather_list[0].get("id", 800)
        entry = self._OWM_ICON_MAP.get(cid)
        if entry:
            label, icon = entry
            if not is_day and cid in (800, 801):
                icon = "🌙"
            return label, icon
        return weather_list[0].get("description", "Unknown").title(), "🌤️"

    def _fetch(self):
        cur_url = (f"{self.CURRENT_URL}?lat={self.lat}&lon={self.lon}"
                   f"&appid={self.api_key}&units=metric")
        fc_url  = (f"{self.FORECAST_URL}?lat={self.lat}&lon={self.lon}"
                   f"&appid={self.api_key}&units=metric&cnt=16")
        self._parse_current(self._fetch_json(cur_url))
        self._parse_forecast(self._fetch_json(fc_url))
        self._last_fetch = time.time()

    def _parse_current(self, raw: Dict):
        ts      = raw.get("dt", time.time())
        sr      = raw.get("sys", {})
        is_day  = sr.get("sunrise", 0) <= ts <= sr.get("sunset", ts + 1)
        wlist   = raw.get("weather", [])
        label, icon = self._owm_condition(wlist, is_day)
        main    = raw.get("main", {}); wind = raw.get("wind", {})
        rain    = raw.get("rain", {})
        rain_mm = float(rain.get("1h", rain.get("3h", 0)) or 0)
        self._current_cache = {
            "temperature":     round(float(main.get("temp",       0)), 1),
            "feels_like":      round(float(main.get("feels_like", 0)), 1),
            "humidity":        round(float(main.get("humidity",   0)), 1),
            "wind_speed":      round(float(wind.get("speed", 0)) * 3.6, 1),
            "wind_direction":  int(wind.get("deg", 0)),
            "rain_mm":         round(rain_mm, 2),
            "rain_intensity":  _rain_to_intensity(rain_mm),
            "condition_label": label, "condition_icon": icon,
            "weather_code":    wlist[0].get("id", 0) if wlist else 0,
            "is_day":          is_day,
            "timestamp":       datetime.utcfromtimestamp(ts).isoformat(),
            "location":        self.name, "error": None,
        }

    def _parse_forecast(self, raw: Dict):
        entries = []
        for item in raw.get("list", []):
            ts = item.get("dt", 0)
            try:
                dt_obj = datetime.utcfromtimestamp(ts)
            except Exception:
                continue
            is_day_f = 6 <= dt_obj.hour < 20
            wlist = item.get("weather", [])
            label, icon = self._owm_condition(wlist, is_day_f)
            main = item.get("main", {}); wind = item.get("wind", {})
            rain_mm = float(item.get("rain", {}).get("3h", 0) or 0)
            entries.append({
                "time":            dt_obj.strftime("%Y-%m-%dT%H:%M"),
                "temperature":     round(float(main.get("temp",     0)), 1),
                "humidity":        round(float(main.get("humidity", 0)), 1),
                "rain_mm":         round(rain_mm, 2),
                "rain_intensity":  _rain_to_intensity(rain_mm),
                "condition_label": label, "condition_icon": icon,
                "wind_speed":      round(float(wind.get("speed", 0)) * 3.6, 1),
                "interval_hours":  3,
            })
        self._forecast_cache = entries


# ─── Google Weather backend ────────────────────────────────────────────────────
class _GoogleBackend:
    """Fetches weather from Google Maps Platform Weather API (true hourly)."""

    CURRENT_URL  = "https://weather.googleapis.com/v1/currentConditions:lookup"
    FORECAST_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"

    def __init__(self, lat, lon, name, api_key, cache_ttl):
        self.lat = lat; self.lon = lon; self.name = name
        self.api_key = api_key; self.cache_ttl = cache_ttl
        self._current_cache:  Optional[Dict]       = None
        self._forecast_cache: Optional[List[Dict]] = None
        self._last_fetch:     float                = 0.0
        self._fetch_error:    Optional[str]        = None

    def maybe_refresh(self):
        if (time.time() - self._last_fetch) < self.cache_ttl:
            return
        try:
            self._fetch(); self._fetch_error = None
        except Exception as exc:
            self._fetch_error = str(exc)
            print(f"[FLOW Weather Google] Fetch failed: {exc}")

    def _build_url(self, base: str, extra: dict = None) -> str:
        params = {"location.latitude": self.lat, "location.longitude": self.lon,
                  "key": self.api_key, "unitsSystem": "METRIC", "languageCode": "en"}
        if extra:
            params.update(extra)
        return f"{base}?{urllib.parse.urlencode(params)}"

    def _fetch_json(self, url: str) -> Dict:
        req = urllib.request.Request(url, headers={"User-Agent": "FLOW-FloodMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _fetch(self):
        cur_raw = self._fetch_json(self._build_url(self.CURRENT_URL))
        fc_raw  = self._fetch_json(self._build_url(self.FORECAST_URL, {"hours": 24}))
        self._parse_current(cur_raw)
        self._parse_forecast(fc_raw)
        self._last_fetch = time.time()

    def _parse_current(self, raw: Dict):
        is_day    = bool(raw.get("isDaytime", True))
        cond      = _google_condition(raw.get("weatherCondition", {}).get("type", "CLEAR"), is_day)
        temp_obj  = raw.get("temperature",         {"degrees": 0})
        feels_obj = raw.get("feelsLikeTemperature", {"degrees": 0})
        wind_obj  = raw.get("wind", {})
        rain_mm   = float(raw.get("precipitation", {}).get("qpf", {"quantity": 0}).get("quantity", 0) or 0)
        ts_raw    = raw.get("currentTime", datetime.utcnow().isoformat() + "Z")
        try:    ts_iso = datetime.fromisoformat(ts_raw.rstrip("Z")).isoformat()
        except: ts_iso = datetime.now().isoformat()
        self._current_cache = {
            "temperature":     round(float(temp_obj.get("degrees",  0)), 1),
            "feels_like":      round(float(feels_obj.get("degrees", 0)), 1),
            "humidity":        round(float(raw.get("relativeHumidity", 0) or 0), 1),
            "wind_speed":      round(float(wind_obj.get("speed", {}).get("value", 0) or 0), 1),
            "wind_direction":  int(wind_obj.get("direction", {}).get("degrees", 0) or 0),
            "rain_mm":         round(rain_mm, 2),
            "rain_intensity":  _rain_to_intensity(rain_mm),
            "condition_label": cond["label"], "condition_icon": cond["icon"],
            "weather_code":    0, "is_day": is_day, "timestamp": ts_iso,
            "location":        self.name, "error": None,
        }

    def _parse_forecast(self, raw: Dict):
        entries = []
        for item in raw.get("forecastHours", []):
            ts_raw = item.get("interval", {}).get("startTime", "")
            try:    dt_obj = datetime.fromisoformat(ts_raw.rstrip("Z"))
            except: continue
            is_day_f = 6 <= dt_obj.hour < 20
            cond     = _google_condition(item.get("weatherCondition", {}).get("type", "CLEAR"), is_day_f)
            wind_obj = item.get("wind", {})
            rain_mm  = float(item.get("precipitation", {}).get("qpf", {"quantity": 0}).get("quantity", 0) or 0)
            entries.append({
                "time":            dt_obj.strftime("%Y-%m-%dT%H:%M"),
                "temperature":     round(float(item.get("temperature", {"degrees": 0}).get("degrees", 0)), 1),
                "humidity":        round(float(item.get("relativeHumidity", 0) or 0), 1),
                "rain_mm":         round(rain_mm, 2),
                "rain_intensity":  _rain_to_intensity(rain_mm),
                "condition_label": cond["label"], "condition_icon": cond["icon"],
                "wind_speed":      round(float(wind_obj.get("speed", {}).get("value", 0) or 0), 1),
                "interval_hours":  1,
            })
        self._forecast_cache = entries


# ─── WeatherService facade ─────────────────────────────────────────────────────
class WeatherService:
    """
    Provider-agnostic weather facade for FLOW.
    Supports:
      • provider="google"         – Google Maps Platform Weather API (true hourly)
      • provider="openweathermap" – OpenWeatherMap API (3-hour intervals)
    """

    PROVIDERS = {
        "google":         ("Google Weather",   "🌐 Google Weather"),
        "openweathermap": ("OpenWeatherMap",   "🌦️ OpenWeatherMap"),
    }

    def __init__(
        self,
        latitude:      float = 6.1248,
        longitude:     float = 100.3673,
        location_name: str   = "Monitoring Site",
        cache_ttl:     int   = 300,
        provider:      str   = WEATHER_PROVIDER,
        api_key:       str   = None,   # deprecated; kept for compat
    ):
        self.latitude      = latitude
        self.longitude     = longitude
        self.location_name = location_name
        self.cache_ttl     = cache_ttl
        self._provider     = provider.lower().strip()
        self._backend      = self._make_backend()

    # ─── Backend factory ──────────────────────────────────────────────────────
    def _make_backend(self):
        if self._provider == "openweathermap":
            return _OWMBackend(self.latitude, self.longitude, self.location_name,
                               OWM_API_KEY, self.cache_ttl)
        return _GoogleBackend(self.latitude, self.longitude, self.location_name,
                              GOOGLE_WEATHER_API_KEY, self.cache_ttl)

    # ─── Provider switching ───────────────────────────────────────────────────
    def set_provider(self, provider: str):
        """Switch provider at runtime; invalidates cache."""
        p = provider.lower().strip()
        if p not in self.PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Valid: {list(self.PROVIDERS)}.")
        if p != self._provider:
            self._provider = p
            self._backend  = self._make_backend()

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def provider_label(self) -> str:
        return self.PROVIDERS.get(self._provider, (self._provider,))[0]

    # ─── Public API ───────────────────────────────────────────────────────────
    def get_current(self) -> Dict:
        """Return current weather as a flat dict (provider-independent)."""
        self._backend.maybe_refresh()
        if self._backend._current_cache:
            return self._backend._current_cache
        return self._error_payload()

    def get_forecast(self, hours: int = 24) -> List[Dict]:
        """
        Return forecast entries for approximately the next `hours` hours.
        Google: true hourly (1-h slots).
        OWM:    3-hour slots — cards are wider in the sidebar UI.
        """
        self._backend.maybe_refresh()
        if self._backend._forecast_cache:
            if self._provider == "openweathermap":
                return self._backend._forecast_cache[:max(1, (hours + 2) // 3)]
            return self._backend._forecast_cache[:hours]
        return []

    def rain_intensity(self) -> float:
        return self.get_current().get("rain_intensity", 0.0)

    @property
    def last_error(self) -> Optional[str]:
        return self._backend._fetch_error

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._backend._last_fetch) > self.cache_ttl * 2

    def force_refresh(self):
        self._backend._last_fetch = 0.0
        self._backend.maybe_refresh()

    def update_location(self, latitude: float, longitude: float, location_name: str):
        changed = (self.latitude != latitude or self.longitude != longitude
                   or self.location_name != location_name)
        self.latitude = latitude; self.longitude = longitude
        self.location_name = location_name
        if changed:
            self._backend = self._make_backend()

    @staticmethod
    def _error_payload() -> Dict:
        return {
            "temperature": 0, "feels_like": 0, "humidity": 0,
            "wind_speed": 0, "wind_direction": 0,
            "rain_mm": 0.0, "rain_intensity": 0.0,
            "condition_label": "Unavailable", "condition_icon": "⚠️",
            "weather_code": -1, "is_day": True,
            "timestamp": datetime.now().isoformat(),
            "location": "Unknown", "error": "Weather data unavailable",
        }


# ─── Streamlit UI helpers ──────────────────────────────────────────────────────

def render_weather_sidebar(ws: WeatherService):
    """
    Call inside a `with st.sidebar:` block to render the compact weather widget.
    Requires `import streamlit as st` in the calling module.
    """
    import streamlit as st

    st.markdown('<div class="sidebar-label">🌤️ LIVE WEATHER</div>', unsafe_allow_html=True)

    # ── Weather API Provider selector ─────────────────────────────────────────
    _provider_options = {
        "🌐 Google Weather":  "google",
        "🌦️ OpenWeatherMap": "openweathermap",
    }
    _prov_labels = list(_provider_options.keys())
    _cur_label   = next((lbl for lbl, key in _provider_options.items() if key == ws.provider),
                        _prov_labels[0])
    _sel_label   = st.selectbox(
        "🔌 Weather API",
        _prov_labels,
        index=_prov_labels.index(_cur_label),
        key="weather_provider_select",
        help="Google Weather: true hourly data. OpenWeatherMap: 3-hour interval forecast.",
    )
    _chosen_prov = _provider_options[_sel_label]
    if _chosen_prov != ws.provider:
        ws.set_provider(_chosen_prov)
        st.rerun()

    # ── Build merged location list ─────────────────────────────────────────────
    custom_locs  = _load_custom_locations()
    preset_locs  = {k: v for k, v in WEATHER_LOCATIONS.items() if k != "📍 Custom Location"}
    all_locations: Dict[str, tuple] = {"📍 Custom Location": None}
    all_locations.update(custom_locs)
    all_locations.update(preset_locs)
    location_names = list(all_locations.keys())

    current_name = ws.location_name
    default_idx  = 0
    for i, name in enumerate(location_names):
        if name == current_name:
            default_idx = i; break

    selected_is_custom_saved = (current_name in custom_locs and current_name in location_names)

    if selected_is_custom_saved:
        sel_col, del_col = st.columns([5, 1])
        with sel_col:
            chosen_name = st.selectbox(
                "📌 Location", location_names, index=default_idx,
                key="weather_location_select",
                help="Choose a preset, a saved custom location, or '📍 Custom Location' to add a new one.",
            )
        with del_col:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("✕", key="weather_delete_btn",
                         help=f"Remove '{current_name}' from saved locations"):
                _delete_custom_location(current_name)
                first_preset = next(iter(preset_locs), None)
                if first_preset:
                    lat, lon = preset_locs[first_preset]
                    ws.update_location(lat, lon, first_preset)
                st.rerun()
    else:
        chosen_name = st.selectbox(
            "📌 Location", location_names, index=default_idx,
            key="weather_location_select",
            help="Choose a preset, a saved custom location, or '📍 Custom Location' to add a new one.",
        )

    coords = all_locations[chosen_name]

    if coords is None:
        if "flow_map_lat" not in st.session_state:
            st.session_state.flow_map_lat = ws.latitude
        if "flow_map_lon" not in st.session_state:
            st.session_state.flow_map_lon = ws.longitude
        if "flow_map_geocoded_for" not in st.session_state:
            st.session_state.flow_map_geocoded_for = None

        _mlat = st.session_state.flow_map_lat
        _mlon = st.session_state.flow_map_lon

        if _FOLIUM_AVAILABLE:
            st.caption("🗺️ Click anywhere on the map to select a location")
            _m = folium.Map(location=[_mlat, _mlon], zoom_start=7, tiles="OpenStreetMap")
            folium.Marker(
                location=[_mlat, _mlon],
                tooltip=f"📍 {_mlat:.4f}°, {_mlon:.4f}°",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(_m)
            _map_data = st_folium(_m, key="flow_location_map", height=310,
                                  returned_objects=["last_clicked"])
            if _map_data and _map_data.get("last_clicked"):
                _click = _map_data["last_clicked"]
                _new_lat = round(float(_click["lat"]), 6)
                _new_lon = round(float(_click["lng"]), 6)
                if abs(_new_lat - _mlat) > 1e-5 or abs(_new_lon - _mlon) > 1e-5:
                    st.session_state.flow_map_lat = _new_lat
                    st.session_state.flow_map_lon = _new_lon
                    st.session_state.flow_map_geocoded_for = None
                    st.rerun()
        else:
            st.info("Install `folium` & `streamlit-folium` to enable map selection.")
            _c1, _c2 = st.columns(2)
            with _c1:
                _new_lat = st.number_input("Latitude",  value=_mlat,
                    min_value=-90.0, max_value=90.0, format="%.6f", key="weather_custom_lat")
            with _c2:
                _new_lon = st.number_input("Longitude", value=_mlon,
                    min_value=-180.0, max_value=180.0, format="%.6f", key="weather_custom_lon")
            if abs(_new_lat - _mlat) > 1e-6 or abs(_new_lon - _mlon) > 1e-6:
                st.session_state.flow_map_lat = _new_lat
                st.session_state.flow_map_lon = _new_lon
                st.session_state.flow_map_geocoded_for = None
                st.rerun()

        _geo_key = (st.session_state.flow_map_lat, st.session_state.flow_map_lon)
        if st.session_state.flow_map_geocoded_for != _geo_key:
            _geocoded = _reverse_geocode(st.session_state.flow_map_lat,
                                         st.session_state.flow_map_lon)
            st.session_state.flow_map_geocoded_for = _geo_key
            st.session_state["weather_custom_name"] = _geocoded

        st.markdown(f"""
<div style="background:rgba(0,180,255,0.07);border:1px solid rgba(0,180,255,0.18);
     border-radius:8px;padding:8px 10px;margin:6px 0 4px;">
  <div style="font-size:10px;color:var(--text-muted);letter-spacing:1px;margin-bottom:6px;">
    📍 SELECTED COORDINATES
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:12px;">
    <div>
      <div style="font-size:10px;color:var(--text-muted);">Latitude</div>
      <div style="font-weight:700;color:var(--accent-cyan);">
        {st.session_state.flow_map_lat:.6f}°
      </div>
    </div>
    <div>
      <div style="font-size:10px;color:var(--text-muted);">Longitude</div>
      <div style="font-weight:700;color:var(--accent-cyan);">
        {st.session_state.flow_map_lon:.6f}°
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        custom_label = st.text_input("Location Name", key="weather_custom_name",
                                     placeholder="e.g. My River Site")

        _btn_apply, _btn_save = st.columns(2)
        with _btn_apply:
            if st.button("🔄 Apply", use_container_width=True, key="weather_apply_btn",
                         help="Use this location now without saving"):
                ws.update_location(st.session_state.flow_map_lat,
                                   st.session_state.flow_map_lon,
                                   custom_label.strip() or "Custom Location")
                st.rerun()
        with _btn_save:
            if st.button("💾 Save", use_container_width=True, key="weather_save_btn",
                         help="Save permanently to the location list"):
                label = custom_label.strip() or "Custom Location"
                locs  = _load_custom_locations()
                locs[label] = (st.session_state.flow_map_lat, st.session_state.flow_map_lon)
                _save_custom_locations(locs)
                ws.update_location(st.session_state.flow_map_lat,
                                   st.session_state.flow_map_lon, label)
                st.rerun()
    else:
        lat, lon = coords
        ws.update_location(lat, lon, chosen_name)

    w   = ws.get_current()
    err = w.get("error") or ws.last_error

    if err and w["weather_code"] == -1:
        st.warning(f"Weather unavailable: {err}")
        return

    # ── Main condition card ────────────────────────────────────────────────────
    rain_color = ("#e74c3c" if w["rain_intensity"] >= 0.75 else
                  "#f39c12" if w["rain_intensity"] >= 0.40 else "#2ecc71")

    st.markdown(f"""
<div style="background:rgba(0,180,255,0.07);border:1px solid rgba(0,180,255,0.18);
     border-radius:10px;padding:12px 14px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span style="font-size:28px;line-height:1;">{w['condition_icon']}</span>
    <div>
      <div style="font-size:13px;font-weight:700;color:var(--text-primary);">
        {w['condition_label']}
      </div>
      <div style="font-size:10px;color:var(--text-muted);">{w['location']}</div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div style="font-size:22px;font-weight:800;color:var(--accent-cyan);">
        {w['temperature']}°C
      </div>
      <div style="font-size:10px;color:var(--text-muted);">Feels {w['feels_like']}°C</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px;color:var(--text-secondary);">
    <div>💧 Humidity: <strong>{w['humidity']}%</strong></div>
    <div>💨 Wind: <strong>{w['wind_speed']} km/h</strong></div>
    <div>🌧️ Rain: <strong>{w['rain_mm']} mm/h</strong></div>
    <div style="color:{rain_color};">⚡ Intensity: <strong>{w['rain_intensity']:.2f}</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 24-HOUR FORECAST — horizontal scrollable with NOW marker ───────────────
    forecast = ws.get_forecast(hours=24)
    if forecast:
        is_owm    = ws.provider == "openweathermap"
        card_w    = "90px" if is_owm else "62px"
        icon_size = "22px" if is_owm else "20px"
        label_h   = "32px" if is_owm else "28px"

        st.markdown(
            '<div style="font-size:10px;color:var(--text-muted);'
            'letter-spacing:1px;margin:8px 0 4px;">24-HOUR FORECAST</div>',
            unsafe_allow_html=True,
        )

        now_hour = datetime.now().hour
        now_idx  = 0
        for _i, _h in enumerate(forecast):
            try:
                if int(_h["time"][-5:].split(":")[0]) == now_hour:
                    now_idx = _i; break
            except Exception:
                pass

        cards_html = ""
        for idx, h in enumerate(forecast):
            hour_label = h["time"][-5:]
            is_now     = (idx == now_idx)

            if is_owm:
                try:
                    hh = int(hour_label.split(":")[0])
                    hour_label = f"{hh:02d}\u2013{(hh+3)%24:02d}h"
                except Exception:
                    pass

            rain_col = ("#e74c3c" if h["rain_intensity"] >= 0.75 else
                        "#f39c12" if h["rain_intensity"] >= 0.40 else
                        "#3498db" if h["rain_mm"] > 0 else "var(--text-muted)")

            if is_now:
                now_marker  = '<div style="font-size:9px;font-weight:800;color:#fff;background:#00b4ff;border-radius:4px;padding:1px 5px;text-align:center;letter-spacing:1px;margin-bottom:4px;">NOW</div>'
                card_border = "border:1.5px solid #00b4ff;background:rgba(0,180,255,0.18);"
                time_color  = "#00d4ff"
                temp_color  = "#00d4ff"
            else:
                now_marker  = ""
                card_border = "border:1px solid rgba(0,180,255,0.12);background:rgba(0,180,255,0.04);"
                time_color  = "var(--text-primary)"
                temp_color  = "var(--accent-cyan)"

            cards_html += f"""
<div style="display:inline-flex;flex-direction:column;align-items:center;
            min-width:{card_w};max-width:{card_w};
            padding:8px 4px 6px;border-radius:8px;
            {card_border}
            margin-right:5px;flex-shrink:0;vertical-align:top;">
  {now_marker}
  <div style="font-size:11px;font-weight:600;color:{time_color};
              margin-bottom:5px;white-space:nowrap;">{hour_label}</div>
  <div style="font-size:{icon_size};line-height:1;margin-bottom:4px;">{h['condition_icon']}</div>
  <div style="font-size:10px;color:var(--text-secondary);text-align:center;
              white-space:normal;line-height:1.2;margin-bottom:5px;
              min-height:{label_h};">{h['condition_label']}</div>
  <div style="font-size:12px;font-weight:700;color:{temp_color};
              margin-bottom:3px;">{h['temperature']}°C</div>
  <div style="font-size:10px;color:{rain_col};">{h['rain_mm']}mm</div>
</div>"""

        st.markdown(f"""
<div style="background:rgba(0,180,255,0.04);border:1px solid rgba(0,180,255,0.15);
            border-radius:8px;overflow-x:auto;overflow-y:hidden;
            padding:10px 8px 8px;white-space:nowrap;
            scrollbar-width:thin;scrollbar-color:rgba(0,180,255,0.3) transparent;">
  {cards_html}
</div>
""", unsafe_allow_html=True)

    if ws.is_stale:
        st.warning("⚠ Weather data is stale — check internet connection.")
    else:
        fetched_at = datetime.fromtimestamp(ws._backend._last_fetch).strftime("%H:%M")
        st.caption(f"Updated {fetched_at} · Refreshes every 5 min · {ws.provider_label}")


def render_weather_main_panel(ws: WeatherService):
    """Render an expanded weather card for the main dashboard area."""
    import streamlit as st

    w        = ws.get_current()
    forecast = ws.get_forecast(hours=24)

    st.markdown("#### 🌤️ Real-Time Weather & Forecast")

    col_a, col_b, col_c, col_d = st.columns(4)
    metrics = [
        (col_a, "🌡️ Temperature",  f"{w['temperature']}°C",  f"Feels {w['feels_like']}°C"),
        (col_b, "💧 Humidity",      f"{w['humidity']}%",       "Relative"),
        (col_c, "💨 Wind Speed",    f"{w['wind_speed']} km/h", f"{w['wind_direction']}° bearing"),
        (col_d, "🌧️ Rain Rate",    f"{w['rain_mm']} mm/h",    f"Intensity {w['rain_intensity']:.2f}"),
    ]
    for col, label, val, delta in metrics:
        with col:
            st.metric(label, val, delta)

    if forecast:
        import pandas as pd
        df = pd.DataFrame(forecast[:24])
        df["hour"] = df["time"].str[-5:]
        df = df.set_index("hour")
        st.line_chart(
            df[["temperature", "rain_mm"]].rename(
                columns={"temperature": "Temp (°C)", "rain_mm": "Rain (mm/h)"}
            ),
            use_container_width=True,
            height=180,
        )
