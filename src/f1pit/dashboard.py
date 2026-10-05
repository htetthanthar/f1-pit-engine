"""Phase 7: data behind the dashboard.

The dashboard never trains or scores a model. It reads a small "bundle" of files exported from the
pipeline's results, so it starts in seconds and can be deployed without the training libraries:

  bundle.json            where the bundle came from, and whether it is demonstration data
  final_metrics.json     the Phase 6 results
  stage2_metrics.json    the validation results (optional)
  laps.parquet           one row per held-out lap: tyre state, each model's pit probability and decision

Everything shown for a lap was computed from laps before it ended (see the leakage rules in the
README), so the race replay shows what a strategist could have seen at that moment.

Run:  python -m f1pit.dashboard            -> app/data/   (the bundle the app shows)
      streamlit run app/streamlit_app.py

This module only needs pandas and numpy, so the deployed app does not install PyTorch or XGBoost.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["RaceID", "Driver", "LapNumber"]
MIN_RACES = 10  # below this, race-level intervals cannot support a conclusion (as in f1pit.evaluate)
CLEAR_DIRECTION = 0.1  # |correlation| a feature needs before its SHAP effect is given a direction
HYPOTHESIS_COLUMNS = [
    "Comparison",
    "Season",
    "Races",
    "Number of races",
    "PR-AUC difference",
    "95% CI low",
    "95% CI high",
    "Bootstrap p",
    "Holm-adjusted p",
    "McNemar p",
    "Reading",
    "key",
]
CONTEXT_COLS = ["Stint", "LapsToThreshold", "Censored", "PrevPosition"]
MODEL_LABELS = {
    "rule_based": "Rule-based",
    "xgboost": "XGBoost",
    "lstm_tyre_aware": "LSTM + Stage 1 (main model)",
    "lstm_no_stage1": "LSTM without Stage 1",
    "bilstm_replica": "Published Bi-LSTM",
    "lstm_tyre_aware_focal": "LSTM + Stage 1, focal loss",
    "lstm_tyre_aware_smote": "LSTM + Stage 1, SMOTE",
}
SPLIT_LABELS = {"valid": "Validation", "test": "Test", "extra_test": "Extra test"}
SUBSET_LABELS = {"excluding_rule_forced": "Without rule-forced races", "all_races": "All dry races"}
HYPOTHESIS_LABELS = {
    "H1_stage1_forecast_helps": "H1: the Stage 1 tyre forecast improves pit prediction",
    "H2_sequence_beats_trees": "H2: the LSTM beats XGBoost on the same inputs",
    "main_model_vs_published_bilstm": "Main model vs the published Bi-LSTM",
}


@dataclass
class Bundle:
    meta: dict
    final: dict
    validation: dict | None
    laps: pd.DataFrame

    @property
    def models(self) -> list[str]:
        return [m for m in MODEL_LABELS if f"{m}_pred" in self.laps.columns]

    @property
    def is_demo(self) -> bool:
        return bool(self.meta.get("demo", False))


# --------------------------------------------------------------------------------------------------
# Writing and reading the bundle
# --------------------------------------------------------------------------------------------------
def export_bundle(
    results_dir: Path, processed_dir: Path, out_dir: Path, demo_note: str | None = None
) -> dict:
    """Copy what the dashboard needs out of the pipeline's results. Returns the bundle description."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    metrics_path = results_dir / "final_metrics.json"
    predictions_path = results_dir / "final_predictions.parquet"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            f"No final results in {results_dir}. Run the final evaluation first: python -m f1pit.evaluate"
        )
    final = json.loads(metrics_path.read_text())
    laps = pd.read_parquet(predictions_path)
    features_path = Path(processed_dir) / "features_stage1.parquet"
    if features_path.exists():
        feats = pd.read_parquet(features_path, columns=KEYS + CONTEXT_COLS)
        laps = laps.merge(feats, on=KEYS, how="left", validate="one_to_one")
        unmatched = int(laps["LapsToThreshold"].isna().sum())
        if unmatched:
            raise RuntimeError(
                f"{unmatched} evaluated laps are not in {features_path}: the feature files changed after "
                "the final evaluation. Run python -m f1pit.evaluate again, then export."
            )
    # the app plots the calibrated probability where there is one, so the raw score is not needed too
    laps = laps.drop(columns=[c[:-11] + "_score" for c in laps.columns if c.endswith("_calibrated")])
    for col in laps.select_dtypes("float64").columns:
        laps[col] = laps[col].astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_fingerprint": final["protocol"].get("models_fingerprint"),
        "evaluation_number": final["protocol"].get("evaluation_number"),
        "demo": demo_note is not None,
        "demo_note": demo_note,
        "laps": int(len(laps)),
        "races": int(laps["RaceID"].nunique()),
    }
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "final_metrics.json").write_text(json.dumps(final))
    validation_path = results_dir / "stage2_metrics.json"
    target = out_dir / "stage2_metrics.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        slim = {"protocol": validation.get("protocol"), "comparisons": validation.get("comparisons", {})}
        slim["experiments"] = {
            name: {"seed_ensemble": e["seed_ensemble"]} for name, e in validation["experiments"].items()
        }
        target.write_text(json.dumps(slim))
    elif target.exists():
        target.unlink()
    laps.to_parquet(out_dir / "laps.parquet", index=False)
    return meta


def load_bundle(path: Path) -> Bundle:
    path = Path(path)
    missing = [f for f in ("bundle.json", "final_metrics.json", "laps.parquet") if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a dashboard bundle (missing {missing}).")
    validation_path = path / "stage2_metrics.json"
    return Bundle(
        meta=json.loads((path / "bundle.json").read_text()),
        final=json.loads((path / "final_metrics.json").read_text()),
        validation=json.loads(validation_path.read_text()) if validation_path.exists() else None,
        laps=pd.read_parquet(path / "laps.parquet"),
    )


def find_bundle(app_dir: Path) -> Path:
    """The bundle to show: $F1PIT_DASHBOARD_DATA, else app/data (your results), else app/demo_data."""
    chosen = os.environ.get("F1PIT_DASHBOARD_DATA")
    if chosen:
        if not (Path(chosen) / "bundle.json").exists():
            raise FileNotFoundError(
                f"F1PIT_DASHBOARD_DATA is set to {chosen}, which holds no dashboard data."
            )
        return Path(chosen)
    for candidate in (Path(app_dir) / "data", Path(app_dir) / "demo_data"):
        if (candidate / "bundle.json").exists():
            return candidate
    raise FileNotFoundError("No dashboard data found. Create it with: python -m f1pit.dashboard")


# --------------------------------------------------------------------------------------------------
# Tables for the results pages
# --------------------------------------------------------------------------------------------------
def available_groups(bundle: Bundle) -> list[tuple[str, str]]:
    """(season group, group of races) pairs that have results, headline group first.

    When a season has no rule-forced races its two groups are identical; only the headline one is listed.
    """
    primary = bundle.final["protocol"]["primary_subset"]
    out = [("valid", "all_races")] if bundle.validation else []
    for split, entry in bundle.final["splits"].items():
        subsets = entry.get("subsets", {})
        usable = [s for s in sorted(subsets, key=lambda s: s != primary) if subsets[s].get("status") == "ok"]
        if any("note" in subsets[s] for s in usable):
            usable = [s for s in usable if s == primary] or usable[:1]
        out += [(split, s) for s in usable]
    return out


def headline_group(bundle: Bundle) -> tuple[str, str, bool] | None:
    """The group the overview shows, and whether it really is the planned headline (test season,
    configured group of races). Falls back to the first held-out group, then to validation."""
    primary = bundle.final["protocol"]["primary_subset"]
    groups = available_groups(bundle)
    if ("test", primary) in groups:
        return "test", primary, True
    held_out = [g for g in groups if g[0] != "valid"]
    chosen = (held_out or groups or [None])[0]
    return (*chosen, False) if chosen else None


def pit_window(bundle: Bundle) -> int:
    """Laps either side of a real stop within which a call counts as finding it."""
    return int(bundle.final["protocol"].get("pit_window_laps", 2))


def notices(bundle: Bundle) -> list[str]:
    """Problems with the results that every page must show."""
    protocol, out = bundle.final["protocol"], []
    if protocol.get("quick_mode"):
        out.append("These models were trained in quick mode, on a few races per season. Not final results.")
    status = (bundle.final.get("integrity") or {}).get("status", "ok")
    if status != "ok":
        out.append(f"The check that the saved models still match the data did not pass ({status}).")
    for name, calibrator in (bundle.final.get("calibrators") or {}).items():
        if calibrator.get("warning"):
            out.append(f"{MODEL_LABELS.get(name, name)}: {calibrator['warning']}.")
    return out


def _experiments(bundle: Bundle, split: str, subset: str) -> dict:
    if split == "valid":
        return {n: e["seed_ensemble"] for n, e in (bundle.validation or {}).get("experiments", {}).items()}
    return bundle.final["splits"][split]["subsets"][subset]["experiments"]


def group_info(bundle: Bundle, split: str, subset: str) -> dict:
    if split == "valid":
        n_races = ((bundle.validation or {}).get("protocol") or {}).get("validation_races")
        few = n_races is not None and n_races < MIN_RACES
        warning = f"only {n_races} races: intervals and p-values are unreliable below {MIN_RACES} races"
        return {"n_races": n_races, "warning": warning if few else None}
    res = bundle.final["splits"][split]["subsets"][subset]
    return {"n_races": res["n_races"], "warning": res.get("warning")}


def model_table(bundle: Bundle, split: str, subset: str) -> pd.DataFrame:
    rows = []
    for name, m in _experiments(bundle, split, subset).items():
        cal = m.get("calibration") or {}
        rows.append(
            {
                "Model": MODEL_LABELS.get(name, name),
                "PR-AUC": m["pr_auc"],
                "95% CI low": m["pr_auc_ci"]["ci_low"],
                "95% CI high": m["pr_auc_ci"]["ci_high"],
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Stop found within window": m["pit_within_window"],
                "Calibration error (after Platt)": cal.get("ece_after_platt"),
                "key": name,
            }
        )
    table = pd.DataFrame(rows)
    column = "Calibration error (after Platt)"
    if not table.empty:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if table[column].isna().all():  # e.g. the validation season, where calibration is not measured
            table = table.drop(columns=column)
    return table


def no_skill_level(bundle: Bundle, split: str, subset: str) -> float | None:
    experiments = _experiments(bundle, split, subset)
    return next(iter(experiments.values()))["no_skill_pr_auc"] if experiments else None


def verdict(difference: dict) -> str:
    """Plain-language reading of a PR-AUC difference and its 95% interval."""
    if difference["ci_low"] > 0:
        return "Supported: the whole interval is above zero"
    if difference["ci_high"] < 0:
        return "Contradicted: the whole interval is below zero"
    return "Not supported: the interval includes zero"


def hypothesis_table(bundle: Bundle) -> pd.DataFrame:
    rows = []

    def add(split: str, subset: str, comparisons: dict, n_races, warning):
        for label, c in comparisons.items():
            d = c["pr_auc_difference"]
            rows.append(
                {
                    "Comparison": HYPOTHESIS_LABELS.get(label, label),
                    "Season": SPLIT_LABELS[split],
                    "Races": SUBSET_LABELS[subset],
                    "Number of races": n_races,
                    "PR-AUC difference": d["estimate"],
                    "95% CI low": d["ci_low"],
                    "95% CI high": d["ci_high"],
                    "Bootstrap p": d["p_not_better"],
                    "Holm-adjusted p": c.get("bootstrap_p_holm"),
                    "McNemar p": c["mcnemar"]["p_value"],
                    "Reading": "Too few races to conclude" if warning else verdict(d),
                    "key": label,
                }
            )

    for split, subset in available_groups(bundle):
        info = group_info(bundle, split, subset)
        if split == "valid":
            comparisons = (bundle.validation or {}).get("comparisons", {})
        else:
            comparisons = bundle.final["splits"][split]["subsets"][subset].get("comparisons", {})
        add(split, subset, comparisons, info["n_races"], info["warning"])
    table = pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)
    table["Holm-adjusted p"] = pd.to_numeric(table["Holm-adjusted p"], errors="coerce")
    return table


def importance_table(bundle: Bundle, split: str, kind: str) -> pd.DataFrame:
    """`kind` is "xgboost_shap", "lstm_tyre_aware_permutation" or "lstm_no_stage1_permutation"."""
    data = bundle.final["splits"].get(split, {}).get("explanations", {}).get(kind)
    if not data:
        return pd.DataFrame()
    if kind == "xgboost_shap":
        rows = [
            {"Feature": f, "Importance": v["mean_abs"], "Direction": _direction(v["direction"])}
            for f, v in data.items()
        ]
    else:
        rows = [
            {
                "Feature": f,
                "Importance": v["pr_auc_drop"],
                "Direction": "Stage 1 input" if _is_stage1(f) else "Other input",
            }
            for f, v in data["features"].items()
        ]
    return pd.DataFrame(rows)


def _direction(value) -> str:
    if value is None or pd.isna(value) or abs(value) < CLEAR_DIRECTION:
        return "No clear direction"
    return "Higher value: more likely to pit" if value > 0 else "Higher value: less likely to pit"


def _is_stage1(feature: str) -> bool:
    return feature.startswith("Stage 1") or feature in ("LapsToThreshold", "Censored")


def calibration_table(bundle: Bundle, split: str, subset: str, model: str) -> pd.DataFrame:
    cal = _experiments(bundle, split, subset).get(model, {}).get("calibration")
    if not cal:
        return pd.DataFrame()
    frames = []
    for key, label in (("curve", "As trained"), ("curve_after_platt", "After Platt scaling")):
        frame = pd.DataFrame(cal[key]).rename(
            columns={"mean_predicted": "Predicted probability", "observed_rate": "Observed pit rate"}
        )
        frame["Version"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Race replay
# --------------------------------------------------------------------------------------------------
def races(bundle: Bundle) -> pd.DataFrame:
    """One row per held-out race, in calendar order."""
    out = (
        bundle.laps.groupby("RaceID", sort=True)
        .agg(
            EventName=("EventName", "first"),
            Split=("Split", "first"),
            RuleForced=("RuleForced", "first"),
            Laps=("LapNumber", "max"),
            Stops=("PitLap", "sum"),
        )
        .reset_index()
    )
    out["Label"] = (
        out["RaceID"].str.slice(0, 4)
        + " "
        + out["EventName"]
        + " (round "
        + out["RaceID"].str.slice(5).str.lstrip("0")
        + ")"
    )
    out.loc[out["RuleForced"].astype(bool), "Label"] += " [rule-forced stops]"
    return out


def probability_column(bundle: Bundle, model: str) -> tuple[str, str]:
    """Column to plot for a model and what it is: a calibrated probability when there is one."""
    if f"{model}_calibrated" in bundle.laps.columns:
        return f"{model}_calibrated", "Pit probability (calibrated)"
    return f"{model}_score", "Model score (not a probability)"


def race_laps(bundle: Bundle, race_id: str) -> pd.DataFrame:
    return bundle.laps[bundle.laps["RaceID"] == race_id].sort_values(["Driver", "LapNumber"])


def pit_wall(bundle: Bundle, race_id: str, lap: int, model: str) -> pd.DataFrame:
    """What the pit wall would see while lap `lap` is being driven: one row per driver still running.

    The recommendation for this lap uses only earlier laps. "Stops so far" counts stops on laps before
    this one, so nothing from the current lap or later is shown.
    """
    laps = race_laps(bundle, race_id)
    column, _ = probability_column(bundle, model)
    value_name = "Pit probability" if column.endswith("_calibrated") else "Score"
    now = laps[laps["LapNumber"] == lap].copy()
    earlier = laps[laps["LapNumber"] < lap]
    stops = earlier.groupby("Driver")["PitLap"].sum()
    now["Stops so far"] = now["Driver"].map(stops).fillna(0).astype(int)
    now["Recommendation"] = np.where(now[f"{model}_pred"] == 1, "PIT THIS LAP", "Stay out")
    table = pd.DataFrame(
        {
            "Position": now["PrevPosition"] if "PrevPosition" in now else np.nan,
            "Driver": now["Driver"],
            "Tyre": now["Compound"].str.title(),
            "Tyre age (laps)": now["TyreLife"].astype(float).round().astype("Int64"),
            "Tyre forecast: laps to pace-loss threshold": _laps_to_threshold(now),
            "Stops so far": now["Stops so far"],
            value_name: now[column].astype(float),
            "Recommendation": now["Recommendation"],
        }
    )
    return table.sort_values(["Position", "Driver"], na_position="last").reset_index(drop=True)


def _laps_to_threshold(frame: pd.DataFrame) -> pd.Series:
    if "LapsToThreshold" not in frame:
        return pd.Series("n/a", index=frame.index)
    laps = frame["LapsToThreshold"].astype(float).round().astype("Int64").astype(str).replace("<NA>", "n/a")
    if "Censored" not in frame:
        return laps
    censored = frame["Censored"].astype(float).fillna(0).astype(bool)
    return laps.where(~censored, laps + "+")


def reveal(bundle: Bundle, race_id: str, lap: int) -> list[str]:
    """Drivers who really did pit at the end of lap `lap` (shown only when the user asks)."""
    laps = race_laps(bundle, race_id)
    return sorted(laps.loc[(laps["LapNumber"] == lap) & (laps["PitLap"] == 1), "Driver"])


def driver_timeline(
    bundle: Bundle, race_id: str, driver: str, model: str, up_to_lap: int | None = None
) -> pd.DataFrame:
    """One driver's race, lap by lap. With `up_to_lap`, only what was known while that lap was being
    driven: laps up to it, and real stops only for laps already finished."""
    laps = race_laps(bundle, race_id)
    one = laps[laps["Driver"] == driver]
    if up_to_lap is not None:
        one = one[one["LapNumber"] <= up_to_lap]
    column, _ = probability_column(bundle, model)
    real = one["PitLap"].fillna(0).astype(int)
    if up_to_lap is not None:
        real = real.where(one["LapNumber"] < up_to_lap, 0)
    return pd.DataFrame(
        {
            "Lap": one["LapNumber"].astype(int),
            "Value": one[column].astype(float),
            "Model says pit": one[f"{model}_pred"].astype(int),
            "Real stop": real,
            "Safety car": one["SafetyCar"].fillna(0).astype(int),
            "Tyre": one["Compound"].astype(str).str.title(),
            "Tyre age": one["TyreLife"].astype(float).round().astype("Int64"),
        }
    ).reset_index(drop=True)


def race_scorecard(bundle: Bundle, race_id: str, model: str, window: int = 2) -> dict:
    """How the model did in one race: real stops, stops it flagged within +/- `window` laps, false alarms."""
    laps = race_laps(bundle, race_id)
    found = total = alarms = false_alarms = 0
    for _, d in laps.groupby("Driver"):
        real = d.loc[d["PitLap"] == 1, "LapNumber"].to_numpy(float)
        flagged = d.loc[d[f"{model}_pred"] == 1, "LapNumber"].to_numpy(float)
        total += len(real)
        alarms += len(flagged)
        found += sum(bool(len(flagged)) and np.abs(flagged - lap).min() <= window for lap in real)
        false_alarms += sum(not len(real) or np.abs(real - lap).min() > window for lap in flagged)
    return {
        "stops": int(total),
        "found": int(found),
        "alarms": int(alarms),
        "false_alarms": int(false_alarms),
    }


def main() -> None:
    from f1pit.config import load_config

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="app/data", help="folder to write the bundle to")
    parser.add_argument("--demo-note", help="mark the bundle as demonstration data, with this explanation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    meta = export_bundle(cfg.results_dir, cfg.processed_dir, Path(args.out), args.demo_note)
    print(f"Dashboard data written to {args.out}: {meta['races']} races, {meta['laps']:,} laps.")
    print("Start the dashboard with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
