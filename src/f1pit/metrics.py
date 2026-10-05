"""Metrics and statistics for pit-lap prediction.

All confidence intervals resample whole RACES, not laps: laps in the same race are not independent,
so resampling laps would make intervals look tighter than they really are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Decision threshold that maximises F1 on the data given (use held-out tuning data only)."""
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def classification_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray, is_probability: bool) -> dict:
    y, score, pred = np.asarray(y).astype(int), np.asarray(score, float), np.asarray(pred).astype(int)
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "no_skill_pr_auc": float(y.mean()),
        "n_laps": int(len(y)),
        "n_pit_laps": int(y.sum()),
    }
    out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1))) if is_probability else None
    return out


def pit_window_hit_rate(meta: pd.DataFrame, pred: np.ndarray, window: int = 2) -> float:
    """Share of real pit stops with a predicted stop within +/- `window` laps for the same driver."""
    df = meta[["RaceID", "Driver", "LapNumber", "PitLap"]].assign(Pred=np.asarray(pred).astype(int))
    hits = []
    for _, d in df.groupby(["RaceID", "Driver"]):
        predicted = d.loc[d["Pred"] == 1, "LapNumber"].to_numpy()
        for lap in d.loc[d["PitLap"] == 1, "LapNumber"]:
            hits.append(bool(len(predicted)) and float(np.abs(predicted - lap).min()) <= window)
    return float(np.mean(hits)) if hits else float("nan")


def _race_index(race_ids: np.ndarray) -> list[np.ndarray]:
    codes, uniques = pd.factorize(np.asarray(race_ids))
    return [np.flatnonzero(codes == k) for k in range(len(uniques))]


def race_bootstrap(
    race_ids, y, score_a, score_b=None, n_samples: int = 1000, seed: int = 42, metric=average_precision_score
) -> dict:
    """95% interval for metric(score_a), or for metric(score_a) - metric(score_b) when score_b is given.

    For a paired comparison it also returns a one-sided bootstrap p-value for "A is better than B":
    (resamples where A did not beat B + 1) / (resamples + 1). The +1 keeps the p-value from ever being
    exactly zero, which a finite number of resamples cannot justify (Phipson and Smyth, 2010).

    With only a handful of races the resamples repeat each other and the interval is too narrow;
    `n_races` is returned so callers can warn about it.
    """
    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, float)
    b = None if score_b is None else np.asarray(score_b, float)
    races = _race_index(race_ids)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_samples):
        idx = np.concatenate([races[k] for k in rng.integers(0, len(races), len(races))])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        value = metric(y[idx], a[idx])
        if b is not None:
            value -= metric(y[idx], b[idx])
        stats.append(value)
    stats = np.array(stats)
    point = metric(y, a) - (metric(y, b) if b is not None else 0.0)
    out = {
        "estimate": float(point),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "n_resamples": int(len(stats)),
        "n_races": int(len(races)),
    }
    if b is not None:
        out["p_not_better"] = float((np.sum(stats <= 0) + 1) / (len(stats) + 1))
    return out


def mcnemar_test(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on the laps where exactly one of the two models is correct."""
    y = np.asarray(y).astype(int)
    a_right = np.asarray(pred_a).astype(int) == y
    b_right = np.asarray(pred_b).astype(int) == y
    only_a, only_b = int(np.sum(a_right & ~b_right)), int(np.sum(~a_right & b_right))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def calibration_table(y, prob, bins: int = 10) -> list[dict]:
    """Reliability table with equal-count bins: mean predicted probability vs observed pit rate.

    Equal-count bins are used because pit laps are rare: with equal-width bins almost every lap would
    fall into the first bin and the rest would hold too few laps to mean anything.
    """
    y, prob = np.asarray(y).astype(int), np.asarray(prob, float)
    order = np.argsort(prob, kind="stable")
    return [
        {"mean_predicted": float(prob[g].mean()), "observed_rate": float(y[g].mean()), "n": int(len(g))}
        for g in np.array_split(order, min(bins, len(order)))
        if len(g)
    ]


def expected_calibration_error(table: list[dict]) -> float:
    """Average gap between predicted probability and observed rate, weighted by bin size (0 = perfect)."""
    total = sum(row["n"] for row in table)
    return float(sum(row["n"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in table) / total)


def _logit(prob) -> np.ndarray:
    p = np.clip(np.asarray(prob, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def fit_platt(y, prob) -> dict:
    """Platt scaling: a logistic curve mapping a model's score to a calibrated probability.

    Fit it on held-out tuning data only. It never changes the ranking of laps, so PR-AUC and ROC-AUC
    are unaffected; only the probability values (Brier score, calibration) change.
    """
    model = LogisticRegression(C=1e6, max_iter=1000).fit(
        _logit(prob).reshape(-1, 1), np.asarray(y).astype(int)
    )
    return {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def apply_platt(params: dict, prob) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(params["slope"] * _logit(prob) + params["intercept"])))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests (controls the chance of any false claim)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m, running, out = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = float(running)
    return out
