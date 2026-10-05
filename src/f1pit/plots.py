"""Charts for the report, drawn from the final-evaluation results (Phase 6).

Every chart is written as a PNG in results/figures/. Charts never compute a result themselves: they
only draw numbers that are already in final_metrics.json or final_predictions.parquet, so a figure in
the report can always be traced back to a saved number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

LABELS = {
    "rule_based": "Rule-based",
    "xgboost": "XGBoost",
    "lstm_tyre_aware": "LSTM + Stage 1 (main)",
    "lstm_no_stage1": "LSTM without Stage 1",
    "bilstm_replica": "Published Bi-LSTM",
    "lstm_tyre_aware_focal": "LSTM + Stage 1, focal loss",
    "lstm_tyre_aware_smote": "LSTM + Stage 1, SMOTE",
}
COLOURS = {
    "rule_based": "#7f7f7f",
    "xgboost": "#ff7f0e",
    "lstm_tyre_aware": "#1f77b4",
    "lstm_no_stage1": "#2ca02c",
    "bilstm_replica": "#9467bd",
    "lstm_tyre_aware_focal": "#17becf",
    "lstm_tyre_aware_smote": "#8c564b",
}
SPLIT_LABELS = {"valid": "Validation", "test": "Test", "extra_test": "Extra test"}
SUBSET_LABELS = {"excluding_rule_forced": "without rule-forced races", "all_races": "all dry races"}
CORE = ["rule_based", "xgboost", "lstm_no_stage1", "lstm_tyre_aware", "bilstm_replica"]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _split_title(results: dict, split: str) -> str:
    seasons = ", ".join(map(str, results["splits"][split]["seasons"]))
    return f"{SPLIT_LABELS[split]} season {seasons}"


def _primary(results: dict, split: str) -> dict | None:
    entry = results["splits"].get(split, {})
    res = entry.get("subsets", {}).get(results["protocol"]["primary_subset"])
    return res if res and res.get("status") == "ok" else None


def plot_model_comparison(results: dict, validation: dict | None, path: Path) -> Path | None:
    """PR-AUC with 95% race-level intervals for every model, on each season."""
    groups = {}
    if validation:
        groups["valid"] = {name: e["seed_ensemble"] for name, e in validation["experiments"].items()}
    for split in results["splits"]:
        res = _primary(results, split)
        if res:
            groups[split] = res["experiments"]
    if not groups:
        return None
    names = [n for n in LABELS if any(n in g for g in groups.values())]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2))
    offsets = np.linspace(-0.25, 0.25, len(groups)) if len(groups) > 1 else [0.0]
    markers = {"valid": "o", "test": "s", "extra_test": "^"}
    shades = {"valid": "#9ecae1", "test": "#08519c", "extra_test": "#d94801"}
    for offset, (split, metrics) in zip(offsets, groups.items(), strict=True):
        ys, xs, lo, hi = [], [], [], []
        for i, name in enumerate(names):
            if name in metrics:
                m = metrics[name]
                ys.append(i + offset)
                xs.append(m["pr_auc"])
                lo.append(m["pr_auc"] - m["pr_auc_ci"]["ci_low"])
                hi.append(m["pr_auc_ci"]["ci_high"] - m["pr_auc"])
        ax.errorbar(
            xs,
            ys,
            xerr=[lo, hi],
            fmt=markers[split],
            color=shades[split],
            capsize=3,
            label=SPLIT_LABELS[split],
        )
        no_skill = next(iter(metrics.values()))["no_skill_pr_auc"]
        ax.axvline(no_skill, color=shades[split], linestyle=":", linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC (higher is better); dotted lines = no-skill level for each season")
    ax.set_title(f"Pit-lap prediction, {SUBSET_LABELS[results['protocol']['primary_subset']]}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(groups), frameon=False)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def _primary_rows(results: dict, predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = predictions[predictions["Split"] == split]
    if results["protocol"]["primary_subset"] == "excluding_rule_forced":
        rows = rows[~rows["RuleForced"].astype(bool)]
    return rows


def plot_pr_curves(results: dict, predictions: pd.DataFrame, split: str, path: Path) -> Path | None:
    res = _primary(results, split)
    if not res:
        return None
    rows = _primary_rows(results, predictions, split)
    y = rows["PitLap"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in CORE:
        if name in res["experiments"] and f"{name}_score" in rows:
            precision, recall, _ = precision_recall_curve(y, rows[f"{name}_score"].to_numpy(float))
            label = f"{LABELS[name]} ({res['experiments'][name]['pr_auc']:.3f})"
            ax.step(recall, precision, where="post", color=COLOURS[name], label=label)
    ax.axhline(y.mean(), color="black", linestyle=":", linewidth=1, label=f"No skill ({y.mean():.3f})")
    ax.set_xlabel("Recall (share of real pit laps found)")
    ax.set_ylabel("Precision (share of predicted pit laps that were real)")
    ax.set_title(f"Precision-recall curves, {_split_title(results, split)}\n(PR-AUC in brackets)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_calibration(results: dict, split: str, path: Path) -> Path | None:
    res = _primary(results, split)
    if not res:
        return None
    names = [n for n in CORE if "calibration" in res["experiments"].get(n, {})]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, key, title in (
        (axes[0], "curve", "As trained"),
        (axes[1], "curve_after_platt", "After Platt scaling (fitted on held-out training races)"),
    ):
        top = 0.0
        for name in names:
            cal = res["experiments"][name]["calibration"]
            x = [row["mean_predicted"] for row in cal[key]]
            y = [row["observed_rate"] for row in cal[key]]
            ece = cal["ece"] if key == "curve" else cal["ece_after_platt"]
            ax.plot(
                x, y, marker="o", markersize=4, color=COLOURS[name], label=f"{LABELS[name]} (ECE {ece:.3f})"
            )
            top = max(top, max(x), max(y))
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0, min(1.0, top * 1.05 + 0.01) if key == "curve_after_platt" else 1.0)
        ax.set_xlabel("Mean predicted pit probability (equal-count bins)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed share of pit laps")
    fig.suptitle(f"Calibration, {_split_title(results, split)}")
    return _save(fig, path)


def plot_hypotheses(results: dict, validation: dict | None, path: Path) -> Path | None:
    """PR-AUC differences with 95% intervals: an interval entirely right of zero supports the hypothesis."""
    rows = []
    if validation:
        for label, c in validation.get("comparisons", {}).items():
            rows.append((label, "Validation", c["pr_auc_difference"]))
    for split, entry in results["splits"].items():
        for subset, res in entry.get("subsets", {}).items():
            if res.get("status") != "ok" or "note" in res:
                continue
            for label, c in res["comparisons"].items():
                rows.append(
                    (label, f"{SPLIT_LABELS[split]}, {SUBSET_LABELS[subset]}", c["pr_auc_difference"])
                )
    if not rows:
        return None
    titles = {
        "H1_stage1_forecast_helps": "H1: LSTM with Stage 1 minus LSTM without",
        "H2_sequence_beats_trees": "H2: LSTM with Stage 1 minus XGBoost",
        "main_model_vs_published_bilstm": "Main model minus published Bi-LSTM",
    }
    labels = [k for k in titles if any(r[0] == k for r in rows)]
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 1.2 + 1.5 * len(labels)), sharex=True, squeeze=False)
    for ax, label in zip(axes[:, 0], labels, strict=True):
        mine = [r for r in rows if r[0] == label]
        for i, (_, _where, d) in enumerate(mine):
            colour = "#08519c" if d["ci_low"] > 0 else ("#a50f15" if d["ci_high"] < 0 else "#636363")
            err = [[d["estimate"] - d["ci_low"]], [d["ci_high"] - d["estimate"]]]
            ax.errorbar(d["estimate"], i, xerr=err, fmt="o", color=colour, capsize=3)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(range(len(mine)))
        ax.set_yticklabels([r[1] for r in mine], fontsize=8)
        ax.set_ylim(len(mine) - 0.5, -0.5)
        ax.set_title(titles[label], fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel(
        "PR-AUC difference, 95% race-level bootstrap interval\n(right of zero = first model better)"
    )
    return _save(fig, path)


def _direction_colour(direction) -> str:
    """Grey unless the feature's value and its effect clearly move together (|correlation| >= 0.1)."""
    if direction is None or abs(direction) < 0.1:
        return "#636363"
    return "#cb181d" if direction > 0 else "#2171b5"


def plot_shap(results: dict, split: str, path: Path) -> Path | None:
    shap = results["splits"].get(split, {}).get("explanations", {}).get("xgboost_shap")
    if not shap:
        return None
    names = list(shap)[::-1]
    values = [shap[n]["mean_abs"] for n in names]
    colours = [_direction_colour(shap[n]["direction"]) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(names) + 1.8))
    ax.barh(names, values, color=colours)
    ax.set_xlabel("Mean |SHAP value| (log-odds of a pit stop)")
    ax.set_title(
        f"XGBoost feature contributions, {_split_title(results, split)}\n"
        "red: higher values push towards a stop; blue: away from it; grey: no clear direction",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_permutation(results: dict, split: str, model: str, path: Path) -> Path | None:
    perm = results["splits"].get(split, {}).get("explanations", {}).get(f"{model}_permutation")
    if not perm:
        return None
    names = list(perm["features"])[::-1]
    drops = [perm["features"][n]["pr_auc_drop"] for n in names]
    stds = [perm["features"][n]["std"] for n in names]
    colours = [
        "#08519c" if n.startswith("Stage 1") or n in ("LapsToThreshold", "Censored") else "#9ecae1"
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(names) + 1.8))
    ax.barh(names, drops, xerr=stds, color=colours, capsize=2)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Drop in PR-AUC when the feature is shuffled (baseline {perm['baseline_pr_auc']:.3f})")
    ax.set_title(
        f"{LABELS[model]}: permutation importance, {_split_title(results, split)}\n"
        "dark bars: Stage 1 degradation-forecast inputs",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_example_race(results: dict, predictions: pd.DataFrame, split: str, path: Path) -> Path | None:
    """Predicted pit probability lap by lap for one driver, with the real stops marked."""
    res = _primary(results, split)
    model = "lstm_tyre_aware"
    if not res or f"{model}_score" not in predictions:
        return None
    rows = _primary_rows(results, predictions, split)
    stops = rows.groupby(["RaceID", "Driver"])["PitLap"].agg(["sum", "size"]).reset_index()
    stops = stops[stops["sum"] >= 1].sort_values(
        ["size", "sum", "RaceID", "Driver"], ascending=[False, False, True, True]
    )
    if stops.empty:
        return None
    race, driver = stops.iloc[0][["RaceID", "Driver"]]
    one = rows[(rows["RaceID"] == race) & (rows["Driver"] == driver)].sort_values("LapNumber")
    column = f"{model}_calibrated" if f"{model}_calibrated" in one else f"{model}_score"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(one["LapNumber"], one[column], color=COLOURS[model], label=f"{LABELS[model]}: pit probability")
    flagged = one[one[f"{model}_pred"] == 1]
    ax.scatter(
        flagged["LapNumber"],
        flagged[column],
        color=COLOURS[model],
        zorder=3,
        label="Model says: pit this lap",
    )
    for i, lap in enumerate(one.loc[one["PitLap"] == 1, "LapNumber"]):
        ax.axvline(lap, color="#cb181d", linestyle="--", label="Real pit stop" if i == 0 else None)
    sc = one[one["SafetyCar"].astype(bool)]
    if not sc.empty:
        ax.scatter(
            sc["LapNumber"], np.zeros(len(sc)), marker="|", color="#e6ab02", label="Safety car / VSC lap"
        )
    ax.set_xlabel("Lap")
    ax.set_ylabel("Calibrated pit probability" if column.endswith("calibrated") else "Model score")
    ax.set_title(f"{one['EventName'].iloc[0]} ({race}), driver {driver}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def make_all_figures(
    results: dict, predictions: pd.DataFrame, validation: dict | None, out_dir: Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [
        plot_model_comparison(results, validation, out_dir / "model_comparison.png"),
        plot_hypotheses(results, validation, out_dir / "hypothesis_tests.png"),
    ]
    for split in results["splits"]:
        made += [
            plot_pr_curves(results, predictions, split, out_dir / f"pr_curves_{split}.png"),
            plot_calibration(results, split, out_dir / f"calibration_{split}.png"),
            plot_shap(results, split, out_dir / f"shap_xgboost_{split}.png"),
            plot_permutation(results, split, "lstm_tyre_aware", out_dir / f"permutation_lstm_{split}.png"),
            plot_example_race(results, predictions, split, out_dir / f"example_race_{split}.png"),
        ]
    return [p for p in made if p is not None]
