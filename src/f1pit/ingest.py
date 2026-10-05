"""Phase 1: download race sessions with FastF1 and save one Parquet file per race.

Run:  python -m f1pit.ingest                      (all seasons in the config)
      python -m f1pit.ingest --seasons 2022 2023  (only these seasons)

Races already saved are read from disk, so re-running only fills gaps.

Two sources are used, in this order:

  1. FastF1, which reads the Formula 1 timing service. The service refuses many requests from cloud
     computers (Google Colab, CI runners), so there it often returns nothing.
  2. The TracingInsights archive on GitHub (see `f1pit.archive`): exports of the same FastF1 lap
     tables, readable from anywhere. Used only for races FastF1 could not deliver.

Each saved race records which source it came from (`Source`). `--no-archive` turns the second source
off. If neither source has data for several races in a row, the download stops with an explanation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1pit.config import Config, load_config

log = logging.getLogger("f1pit.ingest")

KEEP_COLS = [
    "Driver",
    "Team",
    "LapNumber",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "Position",
    "IsAccurate",
    "Deleted",
    "FastF1Generated",
]
TIME_COLS = ["LapTime", "Time", "PitInTime", "PitOutTime"]
WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
BOOL_COLS = ["IsAccurate", "Deleted", "FastF1Generated", "FreshTyre", "Rainfall"]
MAX_FAILS_IN_A_ROW = 3


class SourceUnavailableError(RuntimeError):
    """The timing service gave no data for several races in a row."""


class RaceNotAvailableError(ValueError):
    """FastF1 gave nothing and the archive does not hold this race (it answered, the race is not there)."""


def race_path(cfg: Config, year: int, rnd: int) -> Path:
    return cfg.races_dir / f"{year}_{rnd:02d}.parquet"


def session_to_frame(session, year: int, rnd: int) -> pd.DataFrame:
    """Turn a loaded FastF1 race session into one row per lap with plain, Parquet-safe columns."""
    try:
        laps = session.laps
    except Exception as exc:  # FastF1 raises DataNotLoadedError when the download failed
        raise ValueError(f"no lap data: {exc}") from exc
    if laps is None or len(laps) == 0:
        raise ValueError("no lap data returned")

    weather = laps.get_weather_data().reset_index(drop=True)
    laps = pd.DataFrame(laps).reset_index(drop=True)
    if len(weather) != len(laps):
        raise ValueError("weather rows do not line up with lap rows")

    df = pd.DataFrame({c: laps[c] for c in KEEP_COLS if c in laps.columns})
    for c in TIME_COLS:
        df[f"{c}_s"] = pd.to_timedelta(laps[c]).dt.total_seconds() if c in laps.columns else np.nan
    for c in WEATHER_COLS:
        df[c] = weather[c].to_numpy() if c in weather.columns else np.nan
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    df["Compound"] = df["Compound"].astype(str).str.upper()
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    df["Year"] = year
    df["Round"] = rnd
    df["EventName"] = str(session.event["EventName"])
    df["ScheduledLaps"] = _scheduled_laps(session)
    df["Source"] = "fastf1"
    return df


def _scheduled_laps(session) -> float:
    """The race distance as scheduled (FastF1's `total_laps`), or NaN when FastF1 does not have it."""
    try:
        total = session.total_laps
    except Exception:  # FastF1 raises when the lap-count stream was not loaded
        return float("nan")
    return float(total) if total is not None and not pd.isna(total) and total > 0 else float("nan")


def load_race(
    cfg: Config,
    year: int,
    rnd: int,
    fastf1_module=None,
    event_name: str | None = None,
    archive_loader=None,
    use_fastf1: bool = True,
) -> pd.DataFrame:
    """Return one race's laps, downloading it only if it is not saved yet.

    FastF1 is tried first. If it gives nothing and `archive_loader` is set, the race is read from the
    archive instead. The error of the first source is kept in the message when both fail.
    """
    path = race_path(cfg, year, rnd)
    if path.exists():
        return pd.read_parquet(path)
    df, first_error = None, None
    if use_fastf1:
        try:
            if fastf1_module is None:
                import fastf1 as fastf1_module
            session = fastf1_module.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=True)
            df = session_to_frame(session, year, rnd)
        except Exception as exc:
            if archive_loader is None:
                raise
            first_error = exc
    if df is None:
        if archive_loader is None or not event_name:
            raise ValueError(
                f"no lap data ({first_error or 'FastF1 skipped'}) and no archive lookup possible"
            )
        try:
            df = archive_loader(year, rnd, event_name)
        except Exception as exc:
            message = f"FastF1: {first_error or 'skipped'}; archive: {exc}"
            if "not in the archive" in str(exc):
                raise RaceNotAvailableError(message) from exc
            raise ValueError(message) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def event_names(schedule: pd.DataFrame) -> dict[int, str]:
    """Round number -> event name, when the schedule has names."""
    if "EventName" not in schedule.columns:
        return {}
    return {int(r): str(n) for r, n in zip(schedule["RoundNumber"], schedule["EventName"], strict=True)}


def completed_rounds(schedule: pd.DataFrame, now: pd.Timestamp | None = None) -> list[int]:
    """Round numbers of championship races that have already taken place."""
    now = now or pd.Timestamp.now()
    rounds = schedule[schedule["RoundNumber"].astype(int) > 0]
    if "EventDate" in rounds.columns:
        rounds = rounds[pd.to_datetime(rounds["EventDate"]) < now]
    return [int(r) for r in rounds["RoundNumber"]]


def ingest(
    cfg: Config, seasons: list[int] | None = None, fastf1_module=None, archive_loader=None
) -> list[dict]:
    """Download every completed race for the given seasons. Returns the list of failures.

    `archive_loader(year, round, event_name)` is the second source (see `f1pit.archive`); None = FastF1 only.
    """
    if fastf1_module is None:
        import fastf1 as fastf1_module
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.races_dir.mkdir(parents=True, exist_ok=True)
    fastf1_module.Cache.enable_cache(str(cfg.cache_dir))
    if hasattr(fastf1_module, "set_log_level"):  # FastF1 logs every request; keep only real errors
        fastf1_module.set_log_level("ERROR")
        logging.getLogger("fastf1").propagate = False

    failures, fails_in_a_row, blocked = [], 0, False
    fastf1_fails_in_a_row, use_fastf1 = 0, True
    for year in seasons or cfg.all_seasons:
        try:
            schedule = fastf1_module.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            failures.append({"year": year, "round": None, "error": str(exc)[:200]})
            log.warning("SKIP %s schedule: %s", year, exc)
            continue
        rounds = completed_rounds(schedule)
        names = event_names(schedule)
        if cfg.quick:
            rounds = rounds[: cfg.quick_races]
        for rnd in rounds:
            try:
                saved = race_path(cfg, year, rnd).exists()
                df = load_race(cfg, year, rnd, fastf1_module, names.get(rnd), archive_loader, use_fastf1)
                fails_in_a_row = 0
                source = str(df["Source"].iloc[0]) if "Source" in df.columns and len(df) else "fastf1"
                if not saved and use_fastf1:
                    fastf1_fails_in_a_row = 0 if source == "fastf1" else fastf1_fails_in_a_row + 1
                    if fastf1_fails_in_a_row >= MAX_FAILS_IN_A_ROW:
                        use_fastf1 = False
                        log.warning(
                            "FastF1 returned no data %s times in a row; the remaining races are read "
                            "from the archive only.",
                            MAX_FAILS_IN_A_ROW,
                        )
                log.info("OK   %s round %02d%s", year, rnd, " (saved)" if saved else f" (from {source})")
            except RaceNotAvailableError as exc:  # a reachable source says it has no such race: carry on
                failures.append({"year": year, "round": rnd, "error": str(exc)[:200]})
                log.warning(
                    "MISS %s round %02d: not available from FastF1 here and not in the archive", year, rnd
                )
            except Exception as exc:
                fails_in_a_row += 1
                failures.append({"year": year, "round": rnd, "error": str(exc)[:200]})
                log.warning("SKIP %s round %02d: %s", year, rnd, exc)
                if fails_in_a_row >= MAX_FAILS_IN_A_ROW:
                    blocked = True
                    break
        if blocked:
            break

    (cfg.races_dir / "ingest_failures.json").write_text(json.dumps(failures, indent=2))
    if blocked:
        raise SourceUnavailableError(
            f"{MAX_FAILS_IN_A_ROW} races in a row returned no data from any source, so the download was "
            "stopped. The Formula 1 timing service refuses requests from cloud computers such as Google "
            "Colab, and the archive on GitHub does not hold the most recent races. Run "
            "`python -m f1pit.ingest` on your own computer (a home or university connection), then copy "
            f"the files in data/races/ to {cfg.races_dir}. "
            "Races already saved are reused, so running this again only downloads what is missing."
        )
    return failures


def read_all_races(cfg: Config) -> pd.DataFrame:
    files = sorted(cfg.races_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No race files in {cfg.races_dir}. Run `python -m f1pit.ingest` first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def race_sources(cfg: Config) -> pd.DataFrame:
    """Saved races per season and source (races saved before sources were recorded count as fastf1)."""
    rows = []
    for path in sorted(cfg.races_dir.glob("*.parquet")):
        cols = pd.read_parquet(path, columns=None).columns
        source = "fastf1"
        if "Source" in cols:
            values = pd.read_parquet(path, columns=["Source"])["Source"].dropna()
            source = str(values.iloc[0]) if len(values) else "fastf1"
        rows.append({"Year": int(path.stem.split("_")[0]), "Source": source})
    if not rows:
        return pd.DataFrame(columns=["Year", "Source", "Races"])
    return pd.DataFrame(rows).groupby(["Year", "Source"]).size().rename("Races").reset_index()


def print_sources(cfg: Config) -> None:
    table = race_sources(cfg)
    if len(table):
        print("Saved races by season and source:")
        for row in table.itertuples():
            print(f"  {row.Year}: {row.Races} from {row.Source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seasons", type=int, nargs="*")
    parser.add_argument("--no-archive", action="store_true", help="use FastF1 only, never the GitHub archive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loader = None
    if not args.no_archive:
        from f1pit.archive import load_race_from_archive as loader
    cfg = load_config(args.config)
    try:
        failures = ingest(cfg, args.seasons, archive_loader=loader)
    except SourceUnavailableError as error:
        print_sources(cfg)
        raise SystemExit(f"ingest stopped: {error}") from None
    print_sources(cfg)
    print(f"Done. {len(failures)} race(s) could not be downloaded; re-run to retry them.")
    for failure in failures:
        print(f"  missing: {failure['year']} round {failure['round']}")


if __name__ == "__main__":
    main()
