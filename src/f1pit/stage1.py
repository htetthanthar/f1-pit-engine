"""Phase 4: Stage 1 tyre degradation model.

Stage 1 answers one question for every lap: "on current form, how many more laps until this stint's
fuel-corrected pace loss passes the threshold?" That number (`LapsToThreshold`) is Stage 2's extra input.

Two candidate degradation models are compared, as the proposal requires:
  * `linear`  - a straight line of degradation against laps in the stint, one rate per compound
                (and per circuit and compound where a circuit has enough training laps);
  * `xgboost` - gradient-boosted trees using stint age, tyre age, compound, temperatures, traffic and
                circuit, constrained so predicted degradation never falls as the stint gets older.

Projection, for a decision on lap L (only laps up to L-1 are used):
  current = mean of the stint's last 3 clean degradation values up to lap L-1
            (no clean lap yet: the model's expected degradation since the stint started)
  projected(L-1+k) = current + model(L-1+k) - model(L-1),  k = 1..horizon
  LapsToThreshold  = first k with projected >= threshold   (0 if current is already past it;
                     `horizon` and Censored = True if it is never reached within the horizon)

Leakage rules:
  * models are fitted on training races only (rule-forced races excluded);
  * training races' LapsToThreshold comes from models that never saw that race (out-of-fold);
  * the decision lap's own lap time is never used (on a pit lap it already shows the stop).

Run:  python -m f1pit.stage1  -> data/processed/features_stage1.parquet, results/stage1_metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from f1pit.config import Config, load_config

STINT_KEYS = ["RaceID", "Driver", "Stint"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
XGB_INPUTS = [
    "LapsInStint",
    "TyreLife",
    "FreshTyre",
    "IsSoft",
    "IsMedium",
    "IsHard",
    "TrackTemp",
    "AirTemp",
    "GapAhead_s",
    "CircuitCode",
]
# +1 = predicted degradation may only rise with this input; 0 = unconstrained
XGB_MONOTONE = "(1,1,0,0,0,0,0,0,0,0)"


# --------------------------------------------------------------------------------------------------
# Stint state
# --------------------------------------------------------------------------------------------------
def _mean_of_last(values: np.ndarray, n: int) -> np.ndarray:
    """At each position, the mean of the last `n` non-missing values up to and including it."""
    out = np.full(len(values), np.nan)
    window: deque[float] = deque(maxlen=n)
    for i, v in enumerate(values):
        if not np.isnan(v):
            window.append(v)
        if window:
            out[i] = sum(window) / len(window)
    return out


def add_stint_state(df: pd.DataFrame) -> pd.DataFrame:
    """Laps in stint, and smoothed observed degradation (causal and inclusive versions)."""
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)
    stints = [df[k] for k in STINT_KEYS]
    df["LapsInStint"] = df.groupby(STINT_KEYS).cumcount() + 1
    clean_deg = df["Degradation_s"].where(df["CleanLap"].astype(bool))
    # up to and including this lap: used only to MEASURE when the threshold was actually crossed
    df["SmoothedDeg"] = clean_deg.groupby(stints).transform(
        lambda s: pd.Series(_mean_of_last(s.to_numpy(float), 3), index=s.index)
    )
    # up to the PREVIOUS lap: the only version models and Stage 2 may use
    df["ObservedDeg"] = clean_deg.groupby(stints).transform(
        lambda s: pd.Series(_mean_of_last(s.shift(1).to_numpy(float), 3), index=s.index)
    )
    return df


STATE_COLS = [
    "RaceID",
    "LapsInStint",
    "TyreLife",
    "FreshTyre",
    "Compound",
    "EventName",
    "TrackTemp",
    "AirTemp",
    "GapAhead_s",
]


def _state(df: pd.DataFrame, laps_in_stint: pd.Series) -> pd.DataFrame:
    """The same stint seen at a different point in its life (tyre age moves with it)."""
    state = df[STATE_COLS].copy()
    shift = laps_in_stint - df["LapsInStint"]
    state["LapsInStint"] = laps_in_stint
    state["TyreLife"] = df["TyreLife"] + shift
    return state


# --------------------------------------------------------------------------------------------------
# Degradation models
# --------------------------------------------------------------------------------------------------
def _training_laps(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["CleanLap"].astype(bool) & df["Degradation_s"].notna()]


class LinearDegradation:
    """Degradation = intercept + rate x laps in stint, per compound (per circuit and compound when the
    circuit has at least `event_min_laps` clean training laps on that compound)."""

    name = "linear"

    def __init__(self, event_min_laps: int = 200):
        self.event_min_laps = event_min_laps
        self.by_compound: dict[str, tuple[float, float]] = {}
        self.by_event: dict[tuple[str, str], tuple[float, float]] = {}

    @staticmethod
    def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        slope, intercept = np.polyfit(x.astype(float), y.astype(float), 1)
        return float(intercept), float(slope)

    def fit(self, df: pd.DataFrame) -> LinearDegradation:
        laps = _training_laps(df)
        for comp, g in laps.groupby("Compound"):
            if len(g) >= 10:
                self.by_compound[comp] = self._line(
                    g["LapsInStint"].to_numpy(), g["Degradation_s"].to_numpy()
                )
        for (event, comp), g in laps.groupby(["EventName", "Compound"]):
            if len(g) >= self.event_min_laps:
                self.by_event[(event, comp)] = self._line(
                    g["LapsInStint"].to_numpy(), g["Degradation_s"].to_numpy()
                )
        if not self.by_compound:
            raise ValueError("Not enough clean training laps to fit the linear degradation model.")
        return self

    def _coefs(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        fallback = np.mean(list(self.by_compound.values()), axis=0)
        coefs = [
            self.by_event.get((e, c), self.by_compound.get(c, tuple(fallback)))
            for e, c in zip(df["EventName"], df["Compound"], strict=True)
        ]
        arr = np.array(coefs, dtype=float).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        intercept, rate = self._coefs(df)
        return intercept + rate * df["LapsInStint"].to_numpy(float)


class XGBDegradation:
    """Gradient-boosted trees, monotone in stint age and tyre age."""

    name = "xgboost"

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.circuits: dict[str, int] = {}
        self.model: XGBRegressor | None = None

    def _inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in XGB_INPUTS:
            if col == "CircuitCode":
                X[col] = df["EventName"].map(self.circuits).fillna(-1)
            elif col.startswith("Is") and col[2:].upper() in COMPOUNDS:
                X[col] = (df["Compound"] == col[2:].upper()).astype(int)
            else:
                X[col] = df[col].astype(float)
        return X

    def fit(self, df: pd.DataFrame) -> XGBDegradation:
        laps = _training_laps(df)
        self.circuits = {e: i for i, e in enumerate(sorted(laps["EventName"].unique()))}
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            monotone_constraints=XGB_MONOTONE,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.model.fit(self._inputs(laps), laps["Degradation_s"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._inputs(df)).astype(float)


def make_model(name: str, cfg: Config):
    if name == "linear":
        return LinearDegradation(cfg.event_min_laps)
    if name == "xgboost":
        return XGBDegradation(cfg.random_seed)
    raise ValueError(f"unknown Stage 1 model: {name}")


# --------------------------------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------------------------------
def projection(model, df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (current, projected) where projected[:, k-1] is degradation expected k laps after the
    last completed lap. Uses only the decision lap's known state and laps before it."""
    anchor = df["LapsInStint"] - 1
    base = model.predict(_state(df, anchor))
    start = model.predict(_state(df, anchor * 0 + 1))
    current = df["ObservedDeg"].to_numpy(float).copy()
    no_obs = np.isnan(current)
    current[no_obs] = (base - start)[no_obs]
    gains = np.column_stack([model.predict(_state(df, anchor + k)) - base for k in range(1, horizon + 1)])
    return current, current[:, None] + gains


def laps_to_threshold(
    current: np.ndarray, projected: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """First k (1..horizon) with projected >= threshold; 0 if already past; horizon if never (censored)."""
    horizon = projected.shape[1]
    crossed = projected >= threshold
    reached = crossed.any(axis=1)
    laps = np.where(reached, crossed.argmax(axis=1) + 1, horizon).astype(int)
    already = current >= threshold
    laps[already] = 0
    censored = ~reached & ~already
    return laps, censored


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------
def _future(df: pd.DataFrame, column: str, steps: int) -> pd.Series:
    """Value of `column` `steps` laps later in the same stint (NaN past the stint's end)."""
    return df.groupby(STINT_KEYS)[column].shift(-steps)


def forecast_errors(
    df: pd.DataFrame, projected: np.ndarray, current: np.ndarray, horizons: list[int]
) -> dict:
    """Mean absolute error (s) of forecasts of clean-lap degradation h laps after the last completed lap,
    for the model and for a no-change (persistence) reference."""
    clean_deg = df["Degradation_s"].where(df["CleanLap"].astype(bool))
    tmp = df[STINT_KEYS].assign(CleanDeg=clean_deg.to_numpy())
    out = {}
    for h in horizons:
        actual = _future(tmp, "CleanDeg", h - 1).to_numpy(float)  # lap L-1+h
        ok = ~np.isnan(actual) & ~np.isnan(df["ObservedDeg"].to_numpy(float))
        out[f"mae_{h}lap_s"] = float(np.mean(np.abs(projected[ok, h - 1] - actual[ok]))) if ok.any() else None
        out[f"persistence_mae_{h}lap_s"] = (
            float(np.mean(np.abs(current[ok] - actual[ok]))) if ok.any() else None
        )
        out[f"n_{h}lap"] = int(ok.sum())
    return out


def actual_laps_to_threshold(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Laps from the last completed lap until the stint's smoothed degradation first reached the
    threshold. NaN when it never did within the stint (censored) or had already done so."""
    reached = (df["SmoothedDeg"] >= threshold).to_numpy()
    lap = df["LapNumber"].to_numpy(float)
    first = (
        pd.Series(np.where(reached, lap, np.nan), index=df.index)
        .groupby([df[k] for k in STINT_KEYS])
        .transform("min")
    )
    steps = first.to_numpy() - (lap - 1)
    return np.where(steps >= 1, steps, np.nan)


def threshold_errors(df: pd.DataFrame, current: np.ndarray, projected: np.ndarray, threshold: float) -> dict:
    pred, censored = laps_to_threshold(current, projected, threshold)
    actual = actual_laps_to_threshold(df, threshold)
    horizon = projected.shape[1]
    ok = ~np.isnan(actual) & (actual <= horizon) & (current < threshold)
    err = np.abs(pred[ok] - actual[ok])
    stint_ids = df[STINT_KEYS].astype(str).agg("|".join, axis=1)
    crossed = stint_ids[df["SmoothedDeg"] >= threshold].nunique()
    return {
        "threshold_s": threshold,
        "mae_laps": float(err.mean()) if ok.any() else None,
        "within_3_laps": float((err <= 3).mean()) if ok.any() else None,
        "n_laps_scored": int(ok.sum()),
        "stints_reaching_threshold": int(crossed),
        "stints_total": int(stint_ids.nunique()),
        "share_censored_predictions": float(censored.mean()),
    }


def evaluate(model, valid: pd.DataFrame, cfg: Config) -> dict:
    current, projected = projection(model, valid, cfg.horizon_laps)
    result = forecast_errors(valid, projected, current, cfg.forecast_horizons)
    result["thresholds"] = [threshold_errors(valid, current, projected, t) for t in cfg.thresholds_s]
    return result


# --------------------------------------------------------------------------------------------------
# Out-of-fold outputs for Stage 2
# --------------------------------------------------------------------------------------------------
def out_of_fold(make, train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """LapsToThreshold for training laps, each race predicted by a model fitted on the OTHER races."""
    n_races = train["RaceID"].nunique()
    if n_races < 2:
        raise ValueError("Out-of-fold predictions need at least two training races.")
    out = pd.DataFrame(index=train.index, columns=["LapsToThreshold", "Censored"])
    folds = GroupKFold(n_splits=min(cfg.oof_folds, n_races))
    for fit_idx, pred_idx in folds.split(train, groups=train["RaceID"]):
        model = make().fit(train.iloc[fit_idx])
        part = train.iloc[pred_idx]
        current, projected = projection(model, part, cfg.horizon_laps)
        laps, censored = laps_to_threshold(current, projected, cfg.primary_threshold_s)
        out.loc[part.index, "LapsToThreshold"] = laps
        out.loc[part.index, "Censored"] = censored
    return out


def select_model(metrics: dict, model_names, key: str, min_relative_improvement: float) -> str:
    """Keep the simplest model (first name) unless another is better by a clear margin.

    Parsimony: a complex model must earn its place; a rounding-level difference does not count.
    """
    simplest = model_names[0]
    best, best_score = simplest, metrics[simplest][key]
    for name in model_names[1:]:
        score = metrics[name][key]
        if score < metrics[simplest][key] * (1 - min_relative_improvement) and score < best_score:
            best, best_score = name, score
    return best


def run_stage1(
    feats: pd.DataFrame, cfg: Config, model_names=("linear", "xgboost")
) -> tuple[pd.DataFrame, dict]:
    df = add_stint_state(feats)
    fit_mask = (df["Split"] == "train") & ~df["RuleForced"].astype(bool)
    train, valid = df[fit_mask], df[df["Split"].isin(["valid", "validation"])]
    if valid.empty:
        raise ValueError("No validation laps: Stage 1 models are compared on the validation season.")

    metrics = {}
    for name in model_names:
        metrics[name] = evaluate(make_model(name, cfg).fit(train), valid, cfg)
    key = f"mae_{5 if 5 in cfg.forecast_horizons else cfg.forecast_horizons[0]}lap_s"
    best = select_model(metrics, model_names, key, cfg.min_relative_improvement)
    metrics["selected_model"] = best
    metrics["selection_rule"] = (
        f"simplest model ({model_names[0]}) unless another lowers validation {key} "
        f"by at least {cfg.min_relative_improvement:.0%}"
    )

    final = make_model(best, cfg).fit(train)
    current, projected = projection(final, df, cfg.horizon_laps)
    laps, censored = laps_to_threshold(current, projected, cfg.primary_threshold_s)
    df["LapsToThreshold"], df["Censored"] = laps, censored

    oof = out_of_fold(lambda: make_model(best, cfg), train, cfg)
    df.loc[oof.index, "LapsToThreshold"] = oof["LapsToThreshold"].astype(int)
    df.loc[oof.index, "Censored"] = oof["Censored"].astype(bool)
    df["LapsToThreshold"] = df["LapsToThreshold"].astype(int)
    df["Censored"] = df["Censored"].astype(bool)
    df["LapsToThresholdSource"] = np.where(df.index.isin(oof.index), "out_of_fold", "fitted_on_train")
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    feats = pd.read_parquet(cfg.processed_dir / "features.parquet")
    df, metrics = run_stage1(feats, cfg)
    df.to_parquet(cfg.processed_dir / "features_stage1.parquet", index=False)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Validation results ({', '.join(map(str, cfg.valid_seasons))}):")
    for name in ("linear", "xgboost"):
        m = metrics[name]
        line = " | ".join(
            f"{h}-lap MAE {m[f'mae_{h}lap_s']:.3f}s (no-change {m[f'persistence_mae_{h}lap_s']:.3f}s)"
            for h in cfg.forecast_horizons
            if m[f"mae_{h}lap_s"] is not None
        )
        print(f"  {name:8s} {line}")
        for t in m["thresholds"]:
            mae = "n/a" if t["mae_laps"] is None else f"{t['mae_laps']:.2f} laps"
            print(
                f"           threshold {t['threshold_s']:.1f}s: error {mae}, "
                f"{t['stints_reaching_threshold']}/{t['stints_total']} stints reached it"
            )
    print(f"Selected: {metrics['selected_model']} ({metrics['selection_rule']})")


if __name__ == "__main__":
    main()
