"""Phase 8: the report pack.

Writes every table and data fact the written report needs, straight from the saved results, so that no
number is ever typed by hand:

  results/report/data_facts.md      Table 1 (races, laps and pit laps per season) and facts about the run
  results/report/results_tables.md  Tables 2 to 9 (Stage 1, Stage 2, final test, hypotheses, calibration,
                                    explanations, safety car versus green flag)

Nothing is computed here that is not already in the results files, apart from simple counts of the
laps in the feature file. Run it after the final evaluation:

Run:  python -m f1pit.report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1pit.config import Config, load_config

SPLITS = {"train": "Train", "valid": "Validation", "test": "Test", "extra_test": "Extra test"}
SUBSETS = {"excluding_rule_forced": "without rule-forced races", "all_races": "all dry races"}
MODELS = {
    "rule_based": "Rule-based",
    "xgboost": "XGBoost",
    "lstm_tyre_aware": "LSTM + Stage 1 (main)",
    "lstm_no_stage1": "LSTM without Stage 1",
    "bilstm_replica": "Published Bi-LSTM (replica)",
    "lstm_tyre_aware_focal": "LSTM + Stage 1, focal loss",
    "lstm_tyre_aware_smote": "LSTM + Stage 1, SMOTE",
}
COMPARISONS = {
    "H1_stage1_forecast_helps": "H1: LSTM + Stage 1 minus LSTM without Stage 1",
    "H2_sequence_beats_trees": "H2: LSTM + Stage 1 minus XGBoost",
    "main_model_vs_published_bilstm": "LSTM + Stage 1 minus published Bi-LSTM",
}
HYPOTHESES = ("H1_stage1_forecast_helps", "H2_sequence_beats_trees")
TOP_FEATURES = 8


# --------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------
def num(value, digits: int = 3, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = round(float(value), digits) + 0.0  # + 0.0 turns -0.0 into 0.0, so "-0.000" is never printed
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def p_value(value) -> str:
    """Three decimals; a p-value below 0.001 is written as such, never as 0.000."""
    if value is None or pd.isna(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _source_summary(cfg: Config) -> str:
    """How many saved races came from FastF1 and how many from the GitHub archive."""
    from f1pit.ingest import race_sources

    table = race_sources(cfg)
    if not len(table):
        return "no race files found"
    names = {"fastf1": "FastF1", "tracinginsights": "TracingInsights archive (FastF1 exports on GitHub)"}
    totals = table.groupby("Source")["Races"].sum()
    text = "; ".join(f"{int(n)} from {names.get(s, s)}" for s, n in totals.items())
    if "tracinginsights" in totals.index:
        text += ". Archive races have no scheduled distance, so their race length is the laps actually run"
    return text


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------------------------------
# Table 1 and the facts about the run
# --------------------------------------------------------------------------------------------------
def data_facts(cfg: Config, feats: pd.DataFrame, final: dict, log: list, stage1: dict | None) -> str:
    rows = []
    for split, label in SPLITS.items():
        part = feats[feats["Split"] == split]
        if part.empty:
            rows.append([label, "none", 0, 0, 0, 0, "n/a"])
            continue
        seasons = ", ".join(str(int(y)) for y in sorted(part["Year"].unique()))
        forced = part.loc[part["RuleForced"].astype(bool), "RaceID"].nunique()
        pit = int(part["PitLap"].sum())
        rows.append(
            [
                label,
                seasons,
                part["RaceID"].nunique(),
                forced,
                f"{len(part):,}",
                f"{pit:,}",
                f"{pit / len(part):.2%}",
            ]
        )
    headers = ["Split", "Seasons", "Dry races", "Of which rule-forced", "Laps", "Pit laps", "Pit-lap share"]

    kept = set(feats["RaceID"])
    downloaded = sorted(p.stem for p in cfg.races_dir.glob("*.parquet")) if cfg.races_dir.exists() else []
    removed = [r for r in downloaded if r not in kept]
    failures = _read_json(cfg.races_dir / "ingest_failures.json") or []
    forced_ids = sorted(feats.loc[feats["RuleForced"].astype(bool), "RaceID"].unique())
    slopes = _read_json(cfg.processed_dir / "fuel_slopes.json")
    protocol = final["protocol"]

    facts = [
        f"- Report pack generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Models: `{protocol.get('models_fingerprint')}`, trained {protocol.get('models_created')}",
        f"- Evaluations of the test seasons recorded in the log: {len(log)}",
        f"- Races downloaded: {len(downloaded)}; kept after cleaning: {len(kept)}",
        f"- Races removed by cleaning (wet, or no usable dry laps): {', '.join(removed) or 'none'}",
        "- Races that failed to download: "
        + (", ".join(f"{f['year']} round {f['round']}" for f in failures) or "none"),
        "- Source of the saved races: " + _source_summary(cfg),
        f"- Rule-forced races: {', '.join(forced_ids) or 'none'}",
    ]
    if slopes:
        facts.append(
            f"- Fuel slope, median over training races: {slopes['global']:.3f} s per lap "
            f"({len(slopes['per_race'])} training races had enough clean laps)"
        )
    if stage1:
        facts.append(
            f"- Stage 1 model selected on validation: {stage1['selected_model']} ({stage1['selection_rule']})"
        )
    facts.append(f"- Headline group of races: {SUBSETS[protocol['primary_subset']]}")
    return "\n".join(
        [
            "# Data facts",
            "",
            _banner(protocol),
            "## Table 1. Races, laps and pit laps per split (dry races, after cleaning)",
            "",
            table(headers, rows),
            "",
            "## Facts about this run",
            "",
            *facts,
            "",
        ]
    )


def _banner(protocol: dict) -> str:
    if protocol.get("quick_mode"):
        return (
            "> **QUICK MODE.** These models were trained on a few races per season. "
            "Do not use these numbers in the report.\n"
        )
    return ""


# --------------------------------------------------------------------------------------------------
# Tables 2 to 9
# --------------------------------------------------------------------------------------------------
def _stage1_rows(label: str, model: str, m: dict, horizons: list[int], threshold: float) -> list:
    row = [label, model]
    for h in horizons:
        row += [num(m.get(f"mae_{h}lap_s")), num(m.get(f"persistence_mae_{h}lap_s"))]
    at = next((t for t in m.get("thresholds", []) if t["threshold_s"] == threshold), None)
    return row + [num(at["mae_laps"], 2) if at else "n/a"]


def stage1_table(cfg: Config, stage1: dict | None, final: dict) -> str:
    horizons = cfg.forecast_horizons
    headers = ["Season", "Model"]
    for h in horizons:
        headers += [f"{h}-lap error (s)", f"{h}-lap no-change (s)"]
    headers.append(f"Laps-to-{cfg.primary_threshold_s:g} s error (laps)")
    rows = []
    if stage1:
        for name in ("linear", "xgboost"):
            if name in stage1:
                mark = " (selected)" if stage1.get("selected_model") == name else ""
                rows.append(
                    _stage1_rows("Validation", name + mark, stage1[name], horizons, cfg.primary_threshold_s)
                )
    primary = final["protocol"]["primary_subset"]
    for split in ("test", "extra_test"):
        res = final["splits"].get(split, {}).get("subsets", {}).get(primary, {})
        if "stage1" in res:
            model = final["protocol"].get("stage1_model", "selected")
            rows.append(_stage1_rows(SPLITS[split], model, res["stage1"], horizons, cfg.primary_threshold_s))
    note = (
        'Error is the mean absolute error of the degradation forecast. "No-change" is the reference that '
        "assumes no further wear. Held-out seasons use the headline group of races."
    )
    return _section(
        "Table 2. Stage 1 forecast accuracy", table(headers, rows) if rows else "No Stage 1 results.", note
    )


def _model_rows(experiments: dict) -> list[list]:
    rows = []
    for name, m in experiments.items():
        ci = m["pr_auc_ci"]
        rows.append(
            [
                MODELS.get(name, name),
                num(m["pr_auc"]),
                f"{num(ci['ci_low'])} to {num(ci['ci_high'])}",
                num(m["roc_auc"]),
                num(m["f1"]),
                num(m["precision"]),
                num(m["recall"]),
                num(m["pit_within_window"]),
            ]
        )
    return rows


def _model_table(title: str, experiments: dict, n_races, window: int, warning: str | None = None) -> str:
    headers = [
        "Model",
        "PR-AUC",
        "95% CI",
        "ROC-AUC",
        "F1",
        "Precision",
        "Recall",
        f"Stop found within ±{window} laps",
    ]
    no_skill = next(iter(experiments.values()))["no_skill_pr_auc"] if experiments else None
    note = f"{n_races} races. PR-AUC of random guessing: {num(no_skill)}. Intervals resample whole races."
    if warning:
        note += f" WARNING: {warning}."
    return _section(title, table(headers, _model_rows(experiments)), note)


def _section(title: str, body: str, note: str = "") -> str:
    return "\n".join([f"## {title}", "", body, "", note, ""] if note else [f"## {title}", "", body, ""])


def _subset(final: dict, split: str, subset: str) -> dict | None:
    res = final["splits"].get(split, {}).get("subsets", {}).get(subset)
    return res if res and res.get("status") == "ok" else None


def results_tables(cfg: Config, final: dict, validation: dict | None, stage1: dict | None) -> str:
    protocol = final["protocol"]
    primary = protocol["primary_subset"]
    other = next(s for s in SUBSETS if s != primary)
    window = int(protocol.get("pit_window_laps", 2))
    out = ["# Results tables", "", _banner(protocol), stage1_table(cfg, stage1, final)]

    if validation:
        experiments = {n: e["seed_ensemble"] for n, e in validation["experiments"].items()}
        n_valid = (validation.get("protocol") or {}).get("validation_races")
        out.append(
            _model_table("Table 3. Stage 2 models on the validation season", experiments, n_valid, window)
        )
    else:
        out.append(
            _section("Table 3. Stage 2 models on the validation season", "No validation results found.")
        )

    head = _subset(final, "test", primary)
    title = f"Table 4. Final test: test season, {SUBSETS[primary]} (headline)"
    out.append(
        _model_table(title, head["experiments"], head["n_races"], window, head.get("warning"))
        if head
        else _section(title, "No results for this group.")
    )

    parts = []
    for split, subset in (("test", other), ("extra_test", primary), ("extra_test", other)):
        res = _subset(final, split, subset)
        if not res:
            continue
        name = f"{SPLITS[split]} season, {SUBSETS[subset]}"
        if "note" in res:
            parts.append(
                f"**{name}:** no rule-forced races in this season, so it is identical to the table above.\n"
            )
            continue
        part = _model_table(name, res["experiments"], res["n_races"], window, res.get("warning"))
        parts.append("#" + part)  # one heading level below Table 5
    out.append("## Table 5. Final test: sensitivity group and the extra-test season\n")
    out.append("\n".join(parts) if parts else "No further groups.\n")

    out.append(hypothesis_section(final, validation))
    out.append(calibration_section(final, primary))
    out.append(explanation_section(final))
    out.append(track_status_section(final, primary))
    return "\n".join(out)


def hypothesis_section(final: dict, validation: dict | None) -> str:
    headers = [
        "Comparison",
        "Season and races",
        "Races",
        "PR-AUC difference",
        "95% CI",
        "Bootstrap p",
        "Holm p",
        "McNemar p",
        "Reading",
    ]
    rows = []

    def add(where: str, n_races, comparisons: dict, warning):
        for label, c in comparisons.items():
            d = c["pr_auc_difference"]
            rows.append(
                [
                    COMPARISONS.get(label, label),
                    where,
                    n_races,
                    num(d["estimate"], signed=True),
                    f"{num(d['ci_low'], signed=True)} to {num(d['ci_high'], signed=True)}",
                    p_value(d["p_not_better"]),
                    p_value(c.get("bootstrap_p_holm")),
                    p_value(c["mcnemar"]["p_value"]),
                    "too few races to conclude" if warning else reading(d, label in HYPOTHESES),
                ]
            )

    if validation:
        n_valid = (validation.get("protocol") or {}).get("validation_races")
        add("Validation", n_valid, validation.get("comparisons", {}), n_valid is not None and n_valid < 10)
    not_tested = set()
    for split, entry in final["splits"].items():
        for subset, res in entry.get("subsets", {}).items():
            if res.get("status") != "ok" or "note" in res:
                continue
            add(
                f"{SPLITS[split]}, {SUBSETS[subset]}",
                res["n_races"],
                res.get("comparisons", {}),
                res.get("warning"),
            )
            not_tested.update(res.get("hypotheses_not_tested", []))
    rows.sort(key=lambda r: list(COMPARISONS.values()).index(r[0]) if r[0] in COMPARISONS.values() else 99)
    note = (
        "The difference is the first model's PR-AUC minus the second's. The bootstrap p-value is one-sided "
        "(first model better). Holm adjustment covers H1 and H2 within one season and group of races; "
        "it is not applied on validation or to the Bi-LSTM comparison."
    )
    if not_tested:
        missing = ", ".join(COMPARISONS[h] for h in sorted(not_tested))
        note += f" NOT TESTED, because one of the two models was not trained: {missing}."
    return _section("Table 6. Hypothesis tests", table(headers, rows) if rows else "No comparisons.", note)


def reading(difference: dict, is_hypothesis: bool = True) -> str:
    """Plain reading of a difference and its interval. Only H1 and H2 are hypotheses; other comparisons
    are described without the words "supported" or "contradicted"."""
    if difference["ci_low"] > 0:
        return (
            "supported: interval above zero" if is_hypothesis else "first model better: interval above zero"
        )
    if difference["ci_high"] < 0:
        return (
            "contradicted: interval below zero"
            if is_hypothesis
            else "second model better: interval below zero"
        )
    return ("not supported" if is_hypothesis else "no difference shown") + ": interval includes zero"


def calibration_section(final: dict, primary: str) -> str:
    headers = [
        "Season",
        "Model",
        "Brier as trained",
        "Brier after Platt",
        "ECE as trained",
        "ECE after Platt",
    ]
    rows = []
    for split in ("test", "extra_test"):
        res = _subset(final, split, primary)
        for name, m in (res or {}).get("experiments", {}).items():
            cal = m.get("calibration")
            if cal:
                rows.append(
                    [
                        SPLITS[split],
                        MODELS.get(name, name),
                        num(m["brier"]),
                        num(cal["brier_after_platt"]),
                        num(cal["ece"]),
                        num(cal["ece_after_platt"]),
                    ]
                )
    note = (
        "Lower is better. ECE is the expected calibration error with equal-count bins. Platt scaling is "
        "fitted on held-out training races. The rule-based baseline gives no probability, so it is "
        "not listed."
    )
    return _section(
        "Table 7. Calibration (headline group)", table(headers, rows) if rows else "No results.", note
    )


def explanation_section(final: dict) -> str:
    parts = []
    for split in ("test", "extra_test"):
        explanations = final["splits"].get(split, {}).get("explanations") or {}
        shap = explanations.get("xgboost_shap")
        if shap:
            rows = [
                [i + 1, f, num(v["mean_abs"]), num(v["direction"], 2, signed=True)]
                for i, (f, v) in enumerate(list(shap.items())[:TOP_FEATURES])
            ]
            headers = ["Rank", "Feature", "Mean abs. SHAP (log-odds)", "Correlation of value with effect"]
            parts.append(f"**XGBoost, {SPLITS[split]} season: SHAP values**\n\n{table(headers, rows)}\n")
        for model in ("lstm_tyre_aware", "lstm_no_stage1"):
            perm = explanations.get(f"{model}_permutation")
            if perm:
                rows = [
                    [i + 1, f, num(v["pr_auc_drop"]), num(v["std"])]
                    for i, (f, v) in enumerate(list(perm["features"].items())[:TOP_FEATURES])
                ]
                headers = ["Rank", "Feature", "Drop in PR-AUC when shuffled", "Spread over repeats"]
                caption = f"**{MODELS[model]}, {SPLITS[split]} season: permutation importance**"
                parts.append(
                    f"{caption} (baseline PR-AUC {num(perm['baseline_pr_auc'])})\n\n{table(headers, rows)}\n"
                )
    note = (
        f"Top {TOP_FEATURES} features; the full lists are in final_metrics.json. A correlation near zero "
        "means the feature's effect has no clear direction."
    )
    return _section(
        "Table 8. What drives the predictions", "\n".join(parts) if parts else "No explanations.", note
    )


def track_status_section(final: dict, primary: str) -> str:
    headers = ["Season", "Model", "Stops under safety car", "Recall", "Green-flag stops", "Recall"]
    rows = []
    for split in ("test", "extra_test"):
        res = _subset(final, split, primary)
        for name, m in (res or {}).get("experiments", {}).items():
            by = m.get("recall_by_track_status")
            if by:
                rows.append(
                    [
                        SPLITS[split],
                        MODELS.get(name, name),
                        by["safety_car"]["stops"],
                        num(by["safety_car"]["recall"]),
                        by["green_flag"]["stops"],
                        num(by["green_flag"]["recall"]),
                    ]
                )
    note = (
        "Recall is the share of real stops the model called on the exact lap. "
        "Safety car includes virtual safety car."
    )
    return _section(
        "Table 9. Stops under a safety car and under green-flag running (headline group)",
        table(headers, rows) if rows else "No results.",
        note,
    )


# --------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------
def build_report(cfg: Config) -> dict[str, str]:
    """Return {file name: text} for the report pack. Raises if the final evaluation has not run."""
    final = _read_json(cfg.results_dir / "final_metrics.json")
    if final is None:
        raise FileNotFoundError(
            f"No final results in {cfg.results_dir}. Run the final evaluation first: python -m f1pit.evaluate"
        )
    validation = _read_json(cfg.results_dir / "stage2_metrics.json")
    stage1 = _read_json(cfg.results_dir / "stage1_metrics.json")
    log = _read_json(cfg.results_dir / "final_evaluation_log.json") or []
    feats = pd.read_parquet(
        cfg.processed_dir / "features_stage1.parquet",
        columns=["RaceID", "Year", "Split", "PitLap", "RuleForced"],
    )
    return {
        "data_facts.md": data_facts(cfg, feats, final, log, stage1),
        "results_tables.md": results_tables(cfg, final, validation, stage1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg.results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in build_report(cfg).items():
        (out / name).write_text(text)
        print(f"wrote {out / name}")
    print("Charts are in", cfg.results_dir / "figures")


if __name__ == "__main__":
    main()
