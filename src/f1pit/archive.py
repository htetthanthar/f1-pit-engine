"""Second source for race laps: the TracingInsights archive on GitHub.

The archive (https://github.com/TracingInsights-Archive, MIT licence) holds one `laptimes.json` per driver
and race. The files are exports of FastF1's own lap table (the same columns under short names), so they
contain the same real timing data this project normally downloads with FastF1. They are used only when
FastF1 itself returns nothing, which happens on cloud computers such as Google Colab, because the files are
served by GitHub and can be read from anywhere.

Every race loaded from here is marked `Source = "tracinginsights"` so the report can say where each race
came from. The archive has no scheduled race distance, so `ScheduledLaps` is left empty and the cleaning
step uses the laps actually run for those races.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/TracingInsights-Archive/{year}/HEAD/{event}/Race/{name}"
SOURCE_NAME = "tracinginsights"

# archive key -> this project's column
NUMBER_COLS = {
    "lap": "LapNumber",
    "stint": "Stint",
    "life": "TyreLife",
    "pos": "Position",
    "time": "LapTime_s",
    "sesT": "Time_s",
    "pin": "PitInTime_s",
    "pout": "PitOutTime_s",
    "wAT": "AirTemp",
    "wTT": "TrackTemp",
    "wH": "Humidity",
}
BOOL_COLS = {
    "fresh": "FreshTyre",
    "iacc": "IsAccurate",
    "del": "Deleted",
    "ff1G": "FastF1Generated",
    "wR": "Rainfall",
}
REQUIRED = ["lap", "stint", "compound", "life", "pos", "time", "sesT", "pin", "pout", "status", "iacc"]


class ArchiveError(ValueError):
    """The archive does not have this race, or the file is not in the expected form."""


def fetch_json(url: str, tries: int = 3, timeout: int = 30):
    """Download one JSON file. Raises ArchiveError when the file does not exist or cannot be read."""
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ArchiveError(f"not in the archive: {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise ArchiveError(f"could not read {url}: {last}")


def _url(year: int, event: str, name: str) -> str:
    return BASE_URL.format(year=year, event=urllib.parse.quote(event), name=urllib.parse.quote(name))


def _missing(values) -> pd.Series:
    """The archive writes missing values as the text 'None' (and NaN as null)."""
    series = pd.Series(values, dtype=object)
    return series.where(~series.isin(["None", "nan", "NaN", ""]), np.nan)


def driver_frame(driver: str, data: dict) -> pd.DataFrame:
    """One driver's `laptimes.json` as rows in this project's lap format."""
    absent = [k for k in REQUIRED if k not in data]
    if absent:
        raise ArchiveError(f"{driver}: archive file has no {absent}")
    n = len(data["lap"])
    if any(len(data[k]) != n for k in data if isinstance(data[k], list)):
        raise ArchiveError(f"{driver}: columns of different lengths in the archive file")
    out = pd.DataFrame({"Driver": [driver] * n})
    team = _missing(data["team"]) if "team" in data else pd.Series([np.nan] * n, dtype=object)
    out["Team"] = team.to_numpy()
    for key, col in NUMBER_COLS.items():
        out[col] = (
            pd.to_numeric(_missing(data[key]), errors="coerce").to_numpy(float) if key in data else np.nan
        )
    for key, col in BOOL_COLS.items():
        if key in data:
            out[col] = _missing(data[key]).map(lambda v: v is True or v == "True").astype(bool).to_numpy()
        else:
            out[col] = False
    out["Compound"] = _missing(data["compound"]).astype(str).str.upper().to_numpy()
    out["TrackStatus"] = _missing(data["status"]).astype(str).to_numpy()
    return out


def load_race_from_archive(
    year: int, rnd: int, event_name: str, fetch=fetch_json, workers: int = 8
) -> pd.DataFrame:
    """All drivers' laps for one race, in the same columns as `ingest.session_to_frame`."""
    listing = fetch(_url(year, event_name, "drivers.json"))
    drivers = [d["driver"] for d in listing.get("drivers", []) if d.get("driver")]
    if not drivers:
        raise ArchiveError(f"{year} {event_name}: the archive lists no drivers")

    def one(driver: str):
        try:
            return driver, fetch(_url(year, event_name, f"{driver}/laptimes.json"))
        except ArchiveError as exc:
            if "not in the archive" in str(exc):  # a listed driver who did not start has no lap file
                return driver, None
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, drivers))
    frames = [driver_frame(d, data) for d, data in results if data is not None and len(data.get("lap", []))]
    if not frames:
        raise ArchiveError(f"{year} {event_name}: no lap files in the archive")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["LapNumber"]).sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    df["Year"] = year
    df["Round"] = rnd
    df["EventName"] = event_name
    df["ScheduledLaps"] = float("nan")
    df["Source"] = SOURCE_NAME
    order = [
        "Driver",
        "Team",
        "LapNumber",
        "Stint",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "TrackStatus",
        "Position",
    ]
    order += [
        "IsAccurate",
        "Deleted",
        "FastF1Generated",
        "LapTime_s",
        "Time_s",
        "PitInTime_s",
        "PitOutTime_s",
    ]
    order += [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Rainfall",
        "Year",
        "Round",
        "EventName",
        "ScheduledLaps",
        "Source",
    ]
    return df[order]
