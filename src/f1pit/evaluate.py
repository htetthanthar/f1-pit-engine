"""Phase 6: final evaluation on the held-out seasons (test = 2025, extra test = 2026).

No model, decision threshold or setting is fitted or tuned on the test seasons. This script loads
what Stage 2 saved:

  * the trained models (results/models/*.pt, *.json);
  * results/models/manifest.json, which freezes every decision threshold (chosen on the held-out
    training races), the rule-based baseline's stint table and the list of held-out training races.

Two things are fitted here, both from TRAINING races only: the Platt calibration curves (on the
held-out training races) and an identical refit of the selected Stage 1 model (to score its forecasts).

It then scores each held-out season and reports:

  * the Stage 2 metrics with race-level confidence intervals, and the H1 / H2 tests
    (paired race-level bootstrap, McNemar's test, Holm correction across the hypotheses tested);
  * every result twice: without the rule-forced races (the headline, because those stops are set by
    regulation and no model was trained on such races) and with all dry races (a sensitivity check);
  * calibration: reliability curves, expected calibration error and Brier score, as trained and after
    Platt scaling;
  * explanations: exact SHAP values for the XGBoost model and permutation importance for the LSTMs;
  * recall for stops made under a safety car and under green-flag running;
  * the Stage 1 forecast errors on the same seasons.

Safeguards:
  * the saved models must reproduce the Stage 2 validation scores and decisions, otherwise the run
    stops (this catches models that no longer match the feature files);
  * models trained in quick mode are refused unless `--allow-quick` is given;
  * every evaluation is recorded in results/final_evaluation_log.json. If the same trained models
    have already been evaluated on the same held-out laps, the saved results are shown instead
    (tables and charts are redrawn from them); `--rerun` forces a new evaluation. New models or new
    races are evaluated and logged, so the number of looks at the test seasons is always known.

Run:  python -m f1pit.evaluate  -> results/final_metrics.json, final_predictions.parquet,
                                   final_model_table.csv, final_hypothesis_table.csv, figures/*.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from f1pit import stage1
from f1pit.config import Config, load_config
from f1pit.features import FEATURES as BASE_FEATURES
from f1pit.metrics import (
    apply_platt,
    best_f1_threshold,
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    fit_platt,
    holm_adjust,
    mcnemar_test,
    pit_window_hit_rate,
    race_bootstrap,
)
from f1pit.stage2 import (
    COMPARISONS,
    ORDER,
    STAGE1_FEATURES,
    Scaler,
    Stage2Settings,
    _clean_json,
    _json_default,
    _summarise,
    build_bilstm,
    build_lstm,
    make_windows,
    predict_torch,
    prepare,
    rule_based_scores,
)

SUBSETS = ("excluding_rule_forced", "all_races")
SPLITS = ("test", "extra_test")
HYPOTHESES = ("H1_stage1_forecast_helps", "H2_sequence_beats_trees")
META_COLS = ["RaceID", "EventName", "Driver", "LapNumber", "PitLap"]
REPRODUCTION_TOLERANCE = 1e-3
MIN_RACES_FOR_BOOTSTRAP = 10


@dataclass
class EvalSettings:
    primary_subset: str = "excluding_rule_forced"
    calibration_bins: int = 10
    permutation_repeats: int = 3

    @classmethod
    def from_dict(cls, raw: dict) -> EvalSettings:
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
        settings = cls(**raw)
        if settings.primary_subset not in SUBSETS:
            raise ValueError(f"evaluation.primary_subset must be one of {SUBSETS}")
        if settings.calibration_bins < 2 or settings.permutation_repeats < 1:
            raise ValueError("evaluation.calibration_bins must be >= 2 and permutation_repeats >= 1")
        return settings


# --------------------------------------------------------------------------------------------------
# Loading the frozen models
# --------------------------------------------------------------------------------------------------
def load_manifest(model_dir: Path) -> dict:
    path = Path(model_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Stage 2 first: python -m f1pit.stage2")
    return json.loads(path.read_text())


def _rule_tables(entry: dict) -> tuple[pd.Series, pd.Series]:
    rows = entry["rule_tables"]["by_event"]
    by_event = pd.Series(
        [v for _, _, v in rows],
        index=pd.MultiIndex.from_tuples([(e, c) for e, c, _ in rows], names=["EventName", "Compound"]),
        dtype=float,
    )
    return by_event, pd.Series(entry["rule_tables"]["by_compound"], dtype=float)


def load_scorers(name: str, entry: dict, model_dir: Path) -> list:
    """One function per trained seed; each maps a frame of whole races (sorted as `prepare` leaves it)
    to one score per lap. Nothing here can change a model."""
    kind = entry["kind"]
    if kind == "rule":
        by_event, by_compound = _rule_tables(entry)
        return [lambda part: rule_based_scores(part, by_event, by_compound)[0]]

    scorers = []
    for file in entry["files"]:
        path = Path(model_dir) / file
        if not path.exists():
            raise FileNotFoundError(f"{path} is listed in the manifest but missing. Rerun Stage 2.")
        if kind == "xgb":
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(path)
            cols = entry["features"]
            scorers.append(lambda part, m=model, c=cols: m.predict_proba(part[c].astype(float))[:, 1])
        else:
            import torch

            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            settings = Stage2Settings.from_dict(ckpt["settings"])
            scaler = Scaler.from_dict(ckpt["scaler"])
            build = build_bilstm if kind == "bilstm" else build_lstm
            model = build(len(ckpt["features"]) + 1, settings)
            model.load_state_dict(ckpt["state_dict"])
            model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

            def score(part, m=model, sc=scaler, window=settings.window_laps):
                return predict_torch(m, make_windows(part, sc.transform(part), window))

            scorers.append(score)
    return scorers


def score_part(part: pd.DataFrame, scorers: dict, manifest: dict) -> dict:
    """Per-seed scores, the seed-averaged score and the decision at the FROZEN threshold, per model."""
    out = {}
    for name, fns in scorers.items():
        entry = manifest["experiments"][name]
        # same number type and arithmetic as Stage 2, so a lap exactly on a threshold decides the same way
        per_seed = [np.asarray(fn(part)) for fn in fns]
        score = np.mean(per_seed, axis=0)
        if entry["is_probability"]:
            pred = (score >= entry["threshold"]).astype(int)
            per_seed_pred = [
                (s >= t).astype(int) for s, t in zip(per_seed, entry["per_seed_threshold"], strict=True)
            ]
        else:
            by_event, by_compound = _rule_tables(entry)
            pred = rule_based_scores(part, by_event, by_compound)[1]
            per_seed_pred = [pred]
        out[name] = {"per_seed": per_seed, "per_seed_pred": per_seed_pred, "score": score, "pred": pred}
    return out


# --------------------------------------------------------------------------------------------------
# Checks and calibration on data that is NOT the test seasons
# --------------------------------------------------------------------------------------------------
def check_reproduces_validation(valid_scored: dict, valid: pd.DataFrame, saved: pd.DataFrame | None) -> dict:
    """The loaded models must give the validation scores AND decisions Stage 2 wrote; otherwise stop."""
    if saved is None:
        return {"status": "not checked: no Stage 2 validation predictions were given"}
    keys = ["RaceID", "Driver", "LapNumber"]
    merged = valid[keys].merge(saved, on=keys, how="left", validate="one_to_one")
    stale = "The feature files or models have changed since training. Rerun Stage 2, then evaluate."
    diffs, decisions = {}, {}
    for name, scored in valid_scored.items():
        column = f"{name}_score"
        if column not in merged or len(merged) != len(saved) or merged[column].isna().any():
            raise RuntimeError(
                f"The Stage 2 validation predictions do not cover '{name}' on these laps. {stale}"
            )
        diffs[name] = float(np.max(np.abs(merged[column].to_numpy(float) - scored["score"])))
        decisions[name] = int(np.sum(merged[f"{name}_pred"].to_numpy(int) != scored["pred"]))
    bad = {k: v for k, v in diffs.items() if v > REPRODUCTION_TOLERANCE}
    if bad:
        raise RuntimeError(
            f"Saved models do not reproduce the Stage 2 validation predictions (largest differences {bad}). "
            + stale
        )
    changed = {k: v for k, v in decisions.items() if v}
    if changed:
        raise RuntimeError(
            f"Saved models give different validation decisions (laps changed: {changed}). {stale}"
        )
    return {
        "status": "ok",
        "max_abs_difference": diffs,
        "decisions_changed": 0,
        "tolerance": REPRODUCTION_TOLERANCE,
    }


def fit_calibrators(inner_scored: dict, y_inner: np.ndarray, manifest: dict) -> tuple[dict, dict]:
    """Platt scaling per model, fitted on the held-out TRAINING races. Also re-derives each threshold
    from those races as a check on the frozen value (the frozen value is the one that is used)."""
    calibrators, thresholds = {}, {}
    for name, scored in inner_scored.items():
        entry = manifest["experiments"][name]
        if not entry["is_probability"]:
            continue
        calibrators[name] = fit_platt(y_inner, scored["score"])
        if calibrators[name]["slope"] <= 0:
            calibrators[name]["warning"] = "slope is not positive: this calibration reverses the ranking"
        thresholds[name] = {
            "frozen": entry["threshold"],
            "rederived_from_held_out_training_races": best_f1_threshold(y_inner, scored["score"]),
        }
    return calibrators, thresholds


# --------------------------------------------------------------------------------------------------
# Metrics for one group of races
# --------------------------------------------------------------------------------------------------
def _safety_car_recall(y: np.ndarray, pred: np.ndarray, safety_car: np.ndarray) -> dict:
    out = {}
    for label, group in (("safety_car", safety_car), ("green_flag", ~safety_car)):
        stops = (y == 1) & group
        out[label] = {"stops": int(stops.sum()), "recall": float(pred[stops].mean()) if stops.any() else None}
    return out


def evaluate_races(
    part: pd.DataFrame, scored: dict, calibrators: dict, s2: Stage2Settings, es: EvalSettings
) -> dict:
    """All Stage 2 metrics for the laps in `part` (whole races). `scored` must be aligned with `part`."""
    y = part["PitLap"].to_numpy(int)
    out: dict = {
        "races": sorted(part["RaceID"].unique().tolist()),
        "n_races": int(part["RaceID"].nunique()),
        "n_laps": int(len(y)),
        "n_pit_laps": int(y.sum()),
    }
    if y.sum() == 0 or y.sum() == len(y):
        return {**out, "status": "skipped: needs laps with and without a pit stop"}
    out["status"] = "ok"
    if out["n_races"] < MIN_RACES_FOR_BOOTSTRAP:
        out["warning"] = (
            f"only {out['n_races']} races: race-level bootstrap intervals and p-values are unreliable "
            f"below {MIN_RACES_FOR_BOOTSTRAP} races and must not be used for conclusions"
        )
    meta = part[META_COLS].reset_index(drop=True)
    race_ids = meta["RaceID"].to_numpy()
    safety_car = part["SafetyCar"].to_numpy().astype(bool)
    seed = s2.seeds[0]

    out["experiments"] = {}
    for name, sc in scored.items():
        is_prob = name in calibrators
        m = classification_metrics(y, sc["score"], sc["pred"], is_prob)
        m["pit_within_window"] = pit_window_hit_rate(meta, sc["pred"], s2.pit_window_laps)
        m["pr_auc_ci"] = race_bootstrap(race_ids, y, sc["score"], n_samples=s2.bootstrap_samples, seed=seed)
        m["recall_by_track_status"] = _safety_car_recall(y, sc["pred"], safety_car)
        per_seed = []
        for score, pred in zip(sc["per_seed"], sc["per_seed_pred"], strict=True):
            one = classification_metrics(y, score, pred, is_prob)
            one["pit_within_window"] = pit_window_hit_rate(meta, pred, s2.pit_window_laps)
            per_seed.append(one)
        m["per_seed_mean_std"] = _summarise(per_seed)
        if is_prob:
            raw = calibration_table(y, sc["score"], es.calibration_bins)
            calibrated = apply_platt(calibrators[name], sc["score"])
            cal = calibration_table(y, calibrated, es.calibration_bins)
            m["calibration"] = {
                "ece": expected_calibration_error(raw),
                "ece_after_platt": expected_calibration_error(cal),
                "brier_after_platt": float(np.mean((calibrated - y) ** 2)),
                "curve": raw,
                "curve_after_platt": cal,
            }
        out["experiments"][name] = m

    out["comparisons"] = {}
    for label, a, b in COMPARISONS:
        if a in scored and b in scored:
            out["comparisons"][label] = {
                "model_a": a,
                "model_b": b,
                "pr_auc_difference": race_bootstrap(
                    race_ids,
                    y,
                    scored[a]["score"],
                    scored[b]["score"],
                    n_samples=s2.bootstrap_samples,
                    seed=seed,
                ),
                "mcnemar": mcnemar_test(y, scored[a]["pred"], scored[b]["pred"]),
            }
    family = {
        h: out["comparisons"][h]["pr_auc_difference"]["p_not_better"]
        for h in HYPOTHESES
        if h in out["comparisons"]
    }
    for label, p in holm_adjust(family).items():
        out["comparisons"][label]["bootstrap_p_holm"] = p
    out["holm_family"] = list(family)
    out["hypotheses_not_tested"] = [h for h in HYPOTHESES if h not in family]
    return out


def _subset(scored: dict, mask: np.ndarray) -> dict:
    return {
        name: {
            "per_seed": [s[mask] for s in sc["per_seed"]],
            "per_seed_pred": [p[mask] for p in sc["per_seed_pred"]],
            "score": sc["score"][mask],
            "pred": sc["pred"][mask],
        }
        for name, sc in scored.items()
    }


# --------------------------------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------------------------------
def xgboost_shap(entry: dict, part: pd.DataFrame, model_dir: Path) -> dict:
    """Exact SHAP values (TreeSHAP, built into XGBoost) in log-odds, averaged over the trained seeds.

    `mean_abs` is how much a feature moves the prediction on average; `direction` is the correlation
    between the feature's value and its contribution (positive: higher values push towards a stop).
    """
    import xgboost as xgb

    cols = entry["features"]
    X = part[cols].astype(float)
    total = np.zeros((len(X), len(cols)))
    for file in entry["files"]:
        booster = xgb.Booster()
        booster.load_model(Path(model_dir) / file)
        total += booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    contrib = pd.DataFrame(total / len(entry["files"]), columns=cols, index=X.index)
    mean_abs = contrib.abs().mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = {c: float(X[c].corr(contrib[c])) for c in cols}
    order = mean_abs.sort_values(ascending=False).index
    return {c: {"mean_abs": float(mean_abs[c]), "direction": direction[c]} for c in order}


def permutation_importance(
    fns: list, features: list[str], part: pd.DataFrame, repeats: int, seed: int
) -> dict:
    """Drop in PR-AUC when a feature is shuffled across the laps of the held-out races.

    Shuffling breaks the link between the feature and the outcome while keeping its distribution, so a
    large drop means the model relies on it. The two Stage 1 inputs are also shuffled together, which
    measures how much the model relies on the degradation forecast as a whole.
    """
    y = part["PitLap"].to_numpy(int)
    base = float(average_precision_score(y, np.mean([fn(part) for fn in fns], axis=0)))
    groups = {f: [f] for f in features}
    if all(f in features for f in STAGE1_FEATURES):
        groups["Stage 1 forecast (both inputs)"] = list(STAGE1_FEATURES)
    rng = np.random.default_rng(seed)
    out = {}
    for label, cols in groups.items():
        drops = []
        for _ in range(repeats):
            order = rng.permutation(len(part))
            shuffled = part.copy()
            for col in cols:
                shuffled[col] = part[col].to_numpy()[order]
            drops.append(
                base - float(average_precision_score(y, np.mean([fn(shuffled) for fn in fns], axis=0)))
            )
        out[label] = {"pr_auc_drop": float(np.mean(drops)), "std": float(np.std(drops))}
    ranked = dict(sorted(out.items(), key=lambda kv: -kv[1]["pr_auc_drop"]))
    return {"baseline_pr_auc": base, "repeats": repeats, "features": ranked}


# --------------------------------------------------------------------------------------------------
# The final evaluation
# --------------------------------------------------------------------------------------------------
def run_final_evaluation(
    feats: pd.DataFrame,
    cfg: Config,
    model_dir: Path,
    settings: EvalSettings | None = None,
    validation_predictions: pd.DataFrame | None = None,
    stage1_model: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    es = settings or EvalSettings.from_dict(cfg.evaluation)
    manifest = load_manifest(model_dir)
    s2 = Stage2Settings.from_dict(manifest["settings"])
    df = prepare(feats)

    inner = df[df["RaceID"].isin(manifest["inner_validation_races"])]
    missing = set(manifest["inner_validation_races"]) - set(inner["RaceID"])
    if missing or not (inner["Split"] == "train").all():
        raise RuntimeError(
            "The training races have changed since Stage 2 was trained "
            f"(missing or moved: {sorted(missing) or 'split changed'}). Rerun Stage 2, then evaluate."
        )
    scorers = {name: load_scorers(name, entry, model_dir) for name, entry in manifest["experiments"].items()}

    valid = df[df["Split"].isin(["valid", "validation"])]
    integrity = check_reproduces_validation(
        score_part(valid, scorers, manifest), valid, validation_predictions
    )
    calibrators, thresholds = fit_calibrators(
        score_part(inner, scorers, manifest), inner["PitLap"].to_numpy(int), manifest
    )

    results: dict = {
        "protocol": {
            "what": "final evaluation of frozen models; nothing is fitted or tuned on the held-out seasons",
            "models_fingerprint": manifest["fingerprint"],
            "models_created": manifest["created"],
            "quick_mode": bool(manifest.get("quick", False)),
            "primary_subset": es.primary_subset,
            "pit_window_laps": s2.pit_window_laps,
            "window_laps": s2.window_laps,
            "seasons": {
                "train": manifest.get("train_seasons", list(cfg.train_seasons)),
                "valid": manifest.get("valid_seasons", list(cfg.valid_seasons)),
                "test": list(cfg.test_seasons),
                "extra_test": list(cfg.extra_test_seasons),
            },
            "threshold_rule": manifest["threshold_rule"],
            "calibration": "Platt scaling fitted on the held-out training races",
            "confidence_intervals": f"race-level bootstrap, {s2.bootstrap_samples} resamples",
            "multiple_testing": "Holm correction across the hypotheses tested (H1, H2), per season and group",
        },
        "integrity": integrity,
        "thresholds": thresholds,
        "calibrators": calibrators,
        "splits": {},
    }
    stage1_train = df[(df["Split"] == "train") & ~df["RuleForced"].astype(bool)]
    stage1_fitted = None
    if stage1_model is not None and {"LapsInStint", "ObservedDeg", "SmoothedDeg"} <= set(df.columns):
        stage1_fitted = stage1.make_model(stage1_model, cfg).fit(stage1_train)
        results["protocol"]["stage1_model"] = stage1_model

    frames = []
    for split in SPLITS:
        part = df[df["Split"] == split]
        seasons = cfg.test_seasons if split == "test" else cfg.extra_test_seasons
        if part.empty:
            results["splits"][split] = {"seasons": list(seasons), "status": "no races available"}
            continue
        scored = score_part(part, scorers, manifest)
        forced = part["RuleForced"].to_numpy().astype(bool)
        masks = {"excluding_rule_forced": ~forced, "all_races": np.ones(len(part), bool)}
        entry: dict = {
            "seasons": list(seasons),
            "status": "ok",
            "rule_forced_races": sorted(part.loc[forced, "RaceID"].unique().tolist()),
            "subsets": {},
        }
        for subset in SUBSETS:
            mask = masks[subset]
            if subset == "all_races" and not forced.any():
                entry["subsets"][subset] = {
                    **entry["subsets"]["excluding_rule_forced"],
                    "note": "no rule-forced races in this season: identical to excluding_rule_forced",
                }
                continue
            sub = part[mask]
            if sub.empty:
                entry["subsets"][subset] = {"status": "no races available", "n_races": 0}
                continue
            res = evaluate_races(sub, _subset(scored, mask), calibrators, s2, es)
            if stage1_fitted is not None:
                res["stage1"] = stage1.evaluate(stage1_fitted, sub, cfg)
            entry["subsets"][subset] = res

        primary = part[masks[es.primary_subset]]
        explanations = {}
        if not primary.empty and 0 < primary["PitLap"].sum() < len(primary):
            if "xgboost" in manifest["experiments"]:
                explanations["xgboost_shap"] = xgboost_shap(
                    manifest["experiments"]["xgboost"], primary, model_dir
                )
            for name in ("lstm_tyre_aware", "lstm_no_stage1"):
                if name in manifest["experiments"]:
                    explanations[f"{name}_permutation"] = permutation_importance(
                        scorers[name],
                        manifest["experiments"][name]["features"],
                        primary,
                        es.permutation_repeats,
                        s2.seeds[0],
                    )
        entry["explanations"] = explanations
        results["splits"][split] = entry

        frame = part[META_COLS + ["Split", "RuleForced", "SafetyCar", "TyreLife", "Compound"]].reset_index(
            drop=True
        )
        for name, sc in scored.items():
            frame[f"{name}_score"], frame[f"{name}_pred"] = sc["score"], sc["pred"]
            if name in calibrators:
                frame[f"{name}_calibrated"] = apply_platt(calibrators[name], sc["score"])
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return results, predictions


# --------------------------------------------------------------------------------------------------
# Tables, summary and files
# --------------------------------------------------------------------------------------------------
def model_table(results: dict, validation: dict | None = None) -> pd.DataFrame:
    """One row per season, group of races and model: the table for the report."""
    rows = []
    if validation:
        for name, e in validation["experiments"].items():
            m = e["seed_ensemble"]
            rows.append({"split": "valid", "subset": "all_races", "experiment": name, **_row(m)})
    for split, entry in results["splits"].items():
        for subset, res in entry.get("subsets", {}).items():
            for name, m in res.get("experiments", {}).items():
                cal = m.get("calibration", {})
                rows.append(
                    {
                        "split": split,
                        "subset": subset,
                        "experiment": name,
                        **_row(m),
                        "brier_after_platt": cal.get("brier_after_platt"),
                        "ece": cal.get("ece"),
                        "ece_after_platt": cal.get("ece_after_platt"),
                        "n_races": res["n_races"],
                    }
                )
    return pd.DataFrame(rows)


def _row(m: dict) -> dict:
    return {
        "pr_auc": m["pr_auc"],
        "pr_auc_ci_low": m["pr_auc_ci"]["ci_low"],
        "pr_auc_ci_high": m["pr_auc_ci"]["ci_high"],
        "no_skill_pr_auc": m["no_skill_pr_auc"],
        "roc_auc": m["roc_auc"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "pit_within_window": m["pit_within_window"],
        "brier": m["brier"],
        "n_laps": m["n_laps"],
        "n_pit_laps": m["n_pit_laps"],
    }


def hypothesis_table(results: dict, validation: dict | None = None) -> pd.DataFrame:
    rows = []

    def add(split, subset, comparisons):
        for label, c in comparisons.items():
            d = c["pr_auc_difference"]
            rows.append(
                {
                    "split": split,
                    "subset": subset,
                    "comparison": label,
                    "model_a": c["model_a"],
                    "model_b": c["model_b"],
                    "pr_auc_a_minus_b": d["estimate"],
                    "ci_low": d["ci_low"],
                    "ci_high": d["ci_high"],
                    "bootstrap_p": d["p_not_better"],
                    "bootstrap_p_holm": c.get("bootstrap_p_holm"),
                    "mcnemar_p": c["mcnemar"]["p_value"],
                }
            )

    if validation:
        add("valid", "all_races", validation.get("comparisons", {}))
    for split, entry in results["splits"].items():
        for subset, res in entry.get("subsets", {}).items():
            add(split, subset, res.get("comparisons", {}))
    return pd.DataFrame(rows)


def print_summary(results: dict) -> None:
    primary = results["protocol"]["primary_subset"]
    for split, entry in results["splits"].items():
        seasons = ", ".join(map(str, entry["seasons"]))
        if entry["status"] != "ok":
            print(f"\n{split} ({seasons}): {entry['status']}")
            continue
        for subset in (primary, *[s for s in SUBSETS if s != primary]):
            res = entry["subsets"][subset]
            tag = "HEADLINE" if subset == primary else "sensitivity"
            print(f"\n{split} ({seasons}), {subset.replace('_', ' ')} [{tag}]: {res.get('n_races', 0)} races")
            if res.get("status") != "ok":
                print(f"  {res.get('status')}")
                continue
            if "note" in res:
                print(f"  {res['note']}")
                if subset != primary:
                    continue
            if "warning" in res:
                print(f"  WARNING: {res['warning']}")
            print(f"  {'experiment':24s} {'PR-AUC (95% CI)':24s} {'F1':>6s} {'recall':>7s} {'±2 laps':>8s}")
            for name, m in res["experiments"].items():
                ci = m["pr_auc_ci"]
                print(
                    f"  {name:24s} {m['pr_auc']:.3f} ({ci['ci_low']:.3f}-{ci['ci_high']:.3f})    "
                    f"{m['f1']:6.3f} {m['recall']:7.3f} {m['pit_within_window']:8.3f}"
                )
            for label, c in res["comparisons"].items():
                d = c["pr_auc_difference"]
                holm = f", Holm p = {c['bootstrap_p_holm']:.3f}" if "bootstrap_p_holm" in c else ""
                print(
                    f"  {label}: PR-AUC difference {d['estimate']:+.3f} "
                    f"(95% CI {d['ci_low']:+.3f} to {d['ci_high']:+.3f}), "
                    f"bootstrap p {_p_text(d['p_not_better'])}{holm}, "
                    f"McNemar p {_p_text(c['mcnemar']['p_value'])}"
                )
            for label in res.get("hypotheses_not_tested", []):
                print(f"  {label}: NOT TESTED, because one of its two models was not trained in Stage 2")


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def held_out_signature(feats: pd.DataFrame, es: EvalSettings) -> str:
    """Short hash of the held-out laps (features, labels, rule-forced flags) and the headline choice."""
    cols = ORDER + BASE_FEATURES + STAGE1_FEATURES + ["PitLap", "RuleForced", "Split"]
    held = feats.loc[feats["Split"].isin(SPLITS), cols].sort_values(ORDER)
    rows = pd.util.hash_pandas_object(held.astype({"Censored": float}), index=False).to_numpy()
    body = f"{len(held)}|{int(rows.sum(dtype=np.uint64))}|{es.primary_subset}"
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _write_tables_and_figures(results: dict, out: Path, make_figures: bool) -> None:
    validation = _read_json(out / "stage2_metrics.json")
    model_table(results, validation).to_csv(out / "final_model_table.csv", index=False)
    hypothesis_table(results, validation).to_csv(out / "final_hypothesis_table.csv", index=False)
    predictions_path = out / "final_predictions.parquet"
    if make_figures and predictions_path.exists():
        from f1pit.plots import make_all_figures

        make_all_figures(results, pd.read_parquet(predictions_path), validation, out / "figures")


def evaluate_and_save(
    cfg: Config, rerun: bool = False, make_figures: bool = True, allow_quick: bool = False
) -> tuple[dict, str]:
    """Run the final evaluation and write every output. Returns the results and "evaluated" or "skipped"."""
    out = cfg.results_dir
    model_dir = out / "models"
    manifest = load_manifest(model_dir)
    es = EvalSettings.from_dict(cfg.evaluation)
    if manifest.get("quick") and not allow_quick:
        raise RuntimeError(
            "These models were trained in quick mode (a few races per season). Evaluating them would be a "
            "look at the test seasons that cannot support any conclusion. Set `quick: false`, rerun the "
            "pipeline, then evaluate. To try the evaluation step anyway, pass --allow-quick (it is logged)."
        )
    feats = pd.read_parquet(cfg.processed_dir / "features_stage1.parquet")
    signature = held_out_signature(feats, es)
    log_path, metrics_path = out / "final_evaluation_log.json", out / "final_metrics.json"
    log = _read_json(log_path) or []
    done = [
        e
        for e in log
        if e["models_fingerprint"] == manifest["fingerprint"] and e.get("data_signature") == signature
    ]
    if done and not rerun and metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        protocol = saved["protocol"]
        if (protocol["models_fingerprint"], protocol.get("data_signature")) == (
            manifest["fingerprint"],
            signature,
        ):
            print(
                f"These models were already evaluated on these held-out laps ({done[-1]['when']}). "
                "Showing the saved results; pass --rerun to evaluate again (it will be logged)."
            )
            _write_tables_and_figures(saved, out, make_figures)
            return saved, "skipped"

    valid_path = out / "stage2_validation_predictions.parquet"
    if not valid_path.exists():
        raise FileNotFoundError(
            f"{valid_path} not found. It is needed to check that the saved models still match the data. "
            "Rerun Stage 2, then evaluate."
        )
    stage1_metrics = _read_json(out / "stage1_metrics.json")
    results, predictions = run_final_evaluation(
        feats,
        cfg,
        model_dir,
        settings=es,
        validation_predictions=pd.read_parquet(valid_path),
        stage1_model=stage1_metrics["selected_model"] if stage1_metrics else None,
    )
    log.append(
        {
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models_fingerprint": manifest["fingerprint"],
            "data_signature": signature,
            "quick_mode": results["protocol"]["quick_mode"],
            "primary_subset": es.primary_subset,
            "experiments": list(manifest["experiments"]),
            "races": {
                split: entry.get("subsets", {}).get("all_races", {}).get("n_races", 0)
                for split, entry in results["splits"].items()
            },
            "rule_forced_races": {
                split: entry.get("rule_forced_races", []) for split, entry in results["splits"].items()
            },
        }
    )
    results["protocol"]["data_signature"] = signature
    results["protocol"]["evaluation_number"] = len(log)
    results = _clean_json(json.loads(json.dumps(results, default=_json_default)))
    predictions.to_parquet(out / "final_predictions.parquet", index=False)
    metrics_path.write_text(json.dumps(results, indent=2, allow_nan=False))
    log_path.write_text(json.dumps(log, indent=2))
    _write_tables_and_figures(results, out, make_figures)
    return results, "evaluated"


def _p_text(p: float) -> str:
    """'= 0.031' or '< 0.001': a p-value is never printed as 0.000."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--rerun", action="store_true", help="evaluate again even if these models were evaluated"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip the charts")
    parser.add_argument("--allow-quick", action="store_true", help="evaluate models trained in quick mode")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results, status = evaluate_and_save(
        cfg, rerun=args.rerun, make_figures=not args.no_figures, allow_quick=args.allow_quick
    )
    print_summary(results)
    n = results["protocol"].get("evaluation_number", 1)
    if status == "evaluated":
        print(f"\nSaved to {cfg.results_dir}. This was evaluation number {n} of the test seasons.")
        if n > 1:
            print("State in the report how many times the test seasons were evaluated, and why.")


if __name__ == "__main__":
    main()
