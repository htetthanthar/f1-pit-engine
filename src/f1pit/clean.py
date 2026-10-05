"""Phase 2: clean the raw laps and label pit laps.

Run:  python -m f1pit.clean     -> data/processed/laps_clean.parquet
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from f1pit.config import Config, load_config
from f1pit.ingest import read_all_races

DRY = ["SOFT", "MEDIUM", "HARD"]
WET = ["INTERMEDIATE", "WET"]
MAX_PIT_TIME_S = 300.0  # a longer pit visit means the race was suspended or the car was in the garage
FINGERPRINT_COLS = ["Driver", "LapNumber", "Stint", "Compound", "TyreLife", "Position"]


class CopiedRaceError(ValueError):
    """Two race files hold the same laps: one of them is a copy, not a real race."""


def add_race_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RaceID"] = df["Year"].astype(int).astype(str) + "_" + df["Round"].astype(int).astype(str).str.zfill(2)
    return df


def race_fingerprint(race: pd.DataFrame) -> str:
    """The same value for two races only when every driver has the same tyres and position on every lap."""
    cols = [c for c in FINGERPRINT_COLS if c in race.columns]
    part = race[cols].sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    return hashlib.sha1(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes()).hexdigest()


def copied_races(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(copy, original) for every race whose laps are identical to an earlier race.

    Real races never match lap for lap. A match means a race file was duplicated under another season's
    name, which would put the same race in the training and the test data and make every result
    meaningless. Changing only the lap times does not hide a copy, because they are not compared.
    """
    seen: dict[str, str] = {}
    copies = []
    for race_id, race in df.groupby("RaceID", sort=True):
        key = race_fingerprint(race)
        if key in seen:
            copies.append((str(race_id), seen[key]))
        else:
            seen[key] = str(race_id)
    return copies


def drop_wet_races(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove every race in which any driver used intermediate or wet tyres."""
    wet = sorted(df.loc[df["Compound"].isin(WET), "RaceID"].unique())
    return df[~df["RaceID"].isin(wet)].copy(), wet


def label_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """PitLap = 1 for a tyre stop made as a race decision. A pit entry counts only when all of these hold:

      * the driver raced at least one more lap (entering the pits and never coming out is a retirement);
      * the car left on a different set of tyres. If the same compound carries on with its tyre age simply
        counting up, the car only drove through the pit lane: the field was routed through it behind the
        safety car, or the driver served a penalty or had a repair without a tyre change;
      * the time from pit entry to pit exit is at most `MAX_PIT_TIME_S`. Longer means the race was
        suspended (red flag, when every car waits in the pit lane and tyres may be changed freely) or the
        car was repaired in the garage. Neither is a strategy call.

    The excluded entries are kept in `PitLaneNoChange` and `PitSuspended` so they can be counted.
    """
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    by_car = df.groupby(["RaceID", "Driver"])
    has_next_lap = by_car["LapNumber"].shift(-1).notna()
    entered = df["PitInTime_s"].notna() & has_next_lap
    same_set = (by_car["Compound"].shift(-1) == df["Compound"]) & (
        by_car["TyreLife"].shift(-1) == df["TyreLife"] + 1
    )
    pit_time = by_car["PitOutTime_s"].shift(-1) - df["PitInTime_s"]
    suspended = pit_time > MAX_PIT_TIME_S
    df["PitLaneNoChange"] = (entered & same_set).astype(int)
    df["PitSuspended"] = (entered & ~same_set & suspended).astype(int)
    df["PitLap"] = (entered & ~same_set & ~suspended).astype(int)
    return df


def add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    """Race progress and safety-car flags. Track status codes: 4 = safety car, 6/7 = virtual safety car.

    Race length is the SCHEDULED distance, which teams know before the start. The number of laps
    actually run is only known afterwards (a race can be stopped early), so it is used only when the
    scheduled distance is missing or smaller than the laps recorded.
    """
    df = df.copy()
    run = df.groupby("RaceID")["LapNumber"].transform("max")
    if "ScheduledLaps" in df.columns:
        scheduled = df.groupby("RaceID")["ScheduledLaps"].transform("max")
        df["TotalLaps"] = scheduled.where(scheduled >= run, run)
    else:
        df["TotalLaps"] = run
    df["RaceProgress"] = df["LapNumber"] / df["TotalLaps"]
    df["LapsRemaining"] = df["TotalLaps"] - df["LapNumber"]
    df["SafetyCar"] = df["TrackStatus"].astype(str).str.contains("[467]", regex=True).astype(int)
    return df


def flag_rule_forced(df: pd.DataFrame, rule_forced: list[tuple[int, str]]) -> pd.DataFrame:
    df = df.copy()
    df["RuleForced"] = [
        any(y == year and word in str(name) for year, word in rule_forced)
        for y, name in zip(df["Year"], df["EventName"], strict=True)
    ]
    return df


def clean(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Full Phase 2 pipeline. Returns the clean laps and a short report."""
    df = add_race_id(raw)
    copies = copied_races(df)
    if copies:
        listed = ", ".join(f"{copy} (same laps as {original})" for copy, original in copies)
        raise CopiedRaceError(
            f"These race files are copies of another race, not real data: {listed}. Delete them from the "
            "races folder and download the real races (python -m f1pit.ingest). A season must never be "
            "filled with copies of another season."
        )
    races_in = df["RaceID"].nunique()
    df, wet = drop_wet_races(df)
    df = df[df["Compound"].isin(DRY)]
    if "FastF1Generated" in df.columns:
        df = df[~df["FastF1Generated"].astype(bool)]
    df = df.dropna(subset=["LapNumber", "TyreLife", "Stint"])
    df = label_pit_laps(df)
    df = add_race_context(df)
    df = flag_rule_forced(df, cfg.rule_forced_races)
    df["Split"] = [cfg.split_of(int(y)) for y in df["Year"]]
    report = {
        "races_in": int(races_in),
        "wet_races_removed": wet,
        "rule_forced_races": sorted(df.loc[df["RuleForced"], "RaceID"].unique().tolist()),
        "races_out": int(df["RaceID"].nunique()),
        "laps_out": int(len(df)),
        "pit_laps": int(df["PitLap"].sum()),
        "pit_entries_without_tyre_change": int(df["PitLaneNoChange"].sum()),
        "pit_entries_while_suspended": int(df["PitSuspended"].sum()),
        "pit_lap_share": float(np.round(df["PitLap"].mean(), 4)) if len(df) else 0.0,
    }
    return df.reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df, report = clean(read_all_races(cfg), cfg)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.processed_dir / "laps_clean.parquet", index=False)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
