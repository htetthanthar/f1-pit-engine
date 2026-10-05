"""Phase 3: fuel correction, tyre degradation and per-lap model features.

Every feature on lap L uses only information available by the end of lap L-1 (or, for tyre age,
compound and track status, information known during lap L itself). Fuel slopes are estimated from
training races only, then applied unchanged to validation and test races.

Run:  python -m f1pit.features   -> data/processed/features.parquet and fuel_slopes.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from f1pit.config import Config, load_config

STINT_KEYS = ["RaceID", "Driver", "Stint"]
DRIVER_KEYS = ["RaceID", "Driver"]

FEATURES = [
    "TyreLife",
    "IsSoft",
    "IsMedium",
    "IsHard",
    "Stint",
    "RaceProgress",
    "LapsRemaining",
    "SafetyCar",
    "PrevDegradation",
    "AvgDegradation3",
    "PrevPosition",
    "GapAhead_s",
    "AheadPitted",
    "TrackTemp",
    "AirTemp",
]


def clean_lap_mask(df: pd.DataFrame) -> pd.Series:
    """Green-flag, accurately timed, not deleted laps with a lap time."""
    mask = df["IsAccurate"].astype(bool) & (df["SafetyCar"] == 0) & df["LapTime_s"].notna()
    if "Deleted" in df.columns:
        mask &= ~df["Deleted"].astype(bool)
    return mask


def race_fuel_slope(race: pd.DataFrame, min_laps: int = 100) -> float:
    """Seconds per lap from fuel burn and track evolution in one race.

    Regression on clean laps: lap time ~ lap number + tyre age per compound + one baseline per driver
    + one offset per compound. Tyre age resets at each stop while lap number does not, so the two
    effects can be told apart.
    """
    clean = race[clean_lap_mask(race)]
    if len(clean) < min_laps:
        return np.nan
    cols = [clean["LapNumber"].to_numpy(float)]
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        cols.append(np.where(clean["Compound"] == comp, clean["TyreLife"], 0.0))
    X = np.column_stack(
        cols
        + [pd.get_dummies(clean["Driver"]).to_numpy(float), pd.get_dummies(clean["Compound"]).to_numpy(float)]
    )
    coef, *_ = np.linalg.lstsq(X, clean["LapTime_s"].to_numpy(float), rcond=None)
    return float(coef[0])


def fit_fuel_slopes(train: pd.DataFrame, min_laps: int = 100) -> dict:
    """Median slope per circuit over the TRAINING races, plus an overall fallback."""
    per_race = pd.Series(
        {rid: race_fuel_slope(r, min_laps) for rid, r in train.groupby("RaceID")}, dtype=float
    ).dropna()
    if per_race.empty:
        raise ValueError("No training race had enough clean laps to estimate a fuel slope.")
    events = train.drop_duplicates("RaceID").set_index("RaceID")["EventName"]
    by_event = per_race.groupby(events.reindex(per_race.index)).median()
    return {
        "by_event": {str(k): float(v) for k, v in by_event.items()},
        "global": float(per_race.median()),
        "per_race": {str(k): float(v) for k, v in per_race.items()},
    }


def apply_fuel_correction(df: pd.DataFrame, slopes: dict) -> pd.DataFrame:
    df = df.copy()
    df["FuelSlope"] = df["EventName"].map(slopes["by_event"]).fillna(slopes["global"])
    df["LapTimeCorr_s"] = df["LapTime_s"] - df["FuelSlope"] * (df["LapNumber"] - 1)
    return df


def _causal_baseline(values: pd.Series, n_laps: int) -> pd.Series:
    """Running mean of the first `n_laps` clean laps of a stint, using only laps up to the current one."""
    out = np.full(len(values), np.nan)
    total, count = 0.0, 0
    for i, v in enumerate(values.to_numpy(float)):
        if not np.isnan(v) and count < n_laps:
            total += v
            count += 1
        if count:
            out[i] = total / count
    return pd.Series(out, index=values.index)


def add_degradation(df: pd.DataFrame, baseline_laps: int = 3) -> pd.DataFrame:
    """Degradation_s = fuel-corrected lap time minus the stint's baseline known at that lap."""
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    df["CleanLap"] = clean_lap_mask(df)
    clean_times = df["LapTimeCorr_s"].where(df["CleanLap"])
    df["StintBase"] = clean_times.groupby([df[k] for k in STINT_KEYS]).transform(
        lambda s: _causal_baseline(s, baseline_laps)
    )
    df["Degradation_s"] = df["LapTimeCorr_s"] - df["StintBase"]
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous-lap information. The current lap's time and position are never used: on a pit lap
    they already show the stop."""
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    g = df.groupby(DRIVER_KEYS)
    df["PrevDegradation"] = g["Degradation_s"].shift(1)
    df["AvgDegradation3"] = g["Degradation_s"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["PrevPosition"] = g["Position"].shift(1)
    df["PrevTime_s"] = g["Time_s"].shift(1)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        df[f"Is{comp.title()}"] = (df["Compound"] == comp).astype(int)
    return df


def add_car_ahead_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to, and pit status of, the car one place ahead at the end of the previous lap."""
    ahead = (
        df[["RaceID", "LapNumber", "Position", "Time_s", "PitLap"]]
        .dropna(subset=["Position"])
        .rename(
            columns={
                "LapNumber": "PrevLap",
                "Position": "AheadPos",
                "Time_s": "AheadTime_s",
                "PitLap": "AheadPitted",
            }
        )
        .drop_duplicates(["RaceID", "PrevLap", "AheadPos"])
    )
    df = df.assign(PrevLap=df["LapNumber"] - 1, AheadPos=df["PrevPosition"] - 1)
    n_before = len(df)
    df = df.merge(ahead, on=["RaceID", "PrevLap", "AheadPos"], how="left")
    assert len(df) == n_before, "car-ahead merge must not duplicate laps"
    df["GapAhead_s"] = (df["PrevTime_s"] - df["AheadTime_s"]).clip(lower=0, upper=60)
    df["AheadPitted"] = df["AheadPitted"].fillna(0).astype(int)
    return df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, slopes: dict, baseline_laps: int = 3) -> pd.DataFrame:
    df = apply_fuel_correction(df, slopes)
    df = add_degradation(df, baseline_laps)
    df = add_lag_features(df)
    return add_car_ahead_features(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg: Config = load_config(args.config)
    laps = pd.read_parquet(cfg.processed_dir / "laps_clean.parquet")
    train = laps[(laps["Split"] == "train") & ~laps["RuleForced"]]
    slopes = fit_fuel_slopes(train, cfg.min_laps_for_fuel_fit)
    feats = build_features(laps, slopes, cfg.stint_baseline_laps)
    feats.to_parquet(cfg.processed_dir / "features.parquet", index=False)
    (cfg.processed_dir / "fuel_slopes.json").write_text(json.dumps(slopes, indent=2))
    print(f"Fuel slope (training races, median): {slopes['global']:.3f} s/lap")
    print(f"Features: {len(feats):,} laps x {len(FEATURES)} model inputs")


if __name__ == "__main__":
    main()
