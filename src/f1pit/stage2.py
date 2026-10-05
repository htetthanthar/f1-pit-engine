"""Phase 5: Stage 2 pit-decision models and the H1 / H2 experiments.

Every model answers the same question for every lap L: "will this driver pit at the end of lap L?",
using only what was known before the lap ended. Experiments (configs/default.yaml, `stage2.experiments`):

  rule_based             pit when tyre age reaches the circuit's typical stint length for the compound
  xgboost                gradient-boosted trees on the current lap's features, no sequence memory   (H2)
  lstm_tyre_aware        LSTM over the last N laps, base features + Stage 1 forecast    (main model)
  lstm_no_stage1         identical LSTM without the Stage 1 forecast                    (ablation, H1)
  bilstm_replica         published Bi-LSTM (Sasikumar et al., 2025): 3 bidirectional layers,
                         SMOTE + class weights 1:3, no Stage 1 forecast, leakage-free windows
  lstm_tyre_aware_focal  main model trained with focal loss instead of class weights
  lstm_tyre_aware_smote  main model trained on SMOTE-balanced windows instead of class weights

Protocol, identical for every model:
  * fit on training races (rule-forced races excluded);
  * a fixed 20% of training races is held out ("inner validation") for early stopping and for choosing
    the decision threshold, so the validation season is never used to tune anything;
  * scores are reported on the validation season; the test seasons are not touched here (Phase 6);
  * neural models are trained once per seed; per-seed results are averaged, and the seed-averaged
    probability is used for the paired H1 / H2 comparisons.

Run:  python -m f1pit.stage2  -> results/stage2_metrics.json, results/stage2_validation_predictions.parquet,
                                 results/models/ (trained models and manifest.json, which freezes every
                                 decision threshold for the final evaluation in Phase 6)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from f1pit.config import Config, load_config
from f1pit.features import FEATURES as BASE_FEATURES
from f1pit.metrics import (
    best_f1_threshold,
    classification_metrics,
    mcnemar_test,
    pit_window_hit_rate,
    race_bootstrap,
)

STAGE1_FEATURES = ["LapsToThreshold", "Censored"]
ORDER = ["RaceID", "Driver", "LapNumber"]

EXPERIMENTS = {
    "rule_based": {"kind": "rule", "stage1": False, "imbalance": None},
    "xgboost": {"kind": "xgb", "stage1": True, "imbalance": "class_weight"},
    "lstm_tyre_aware": {"kind": "lstm", "stage1": True, "imbalance": "class_weight"},
    "lstm_no_stage1": {"kind": "lstm", "stage1": False, "imbalance": "class_weight"},
    "bilstm_replica": {"kind": "bilstm", "stage1": False, "imbalance": "smote_and_weight"},
    "lstm_tyre_aware_focal": {"kind": "lstm", "stage1": True, "imbalance": "focal"},
    "lstm_tyre_aware_smote": {"kind": "lstm", "stage1": True, "imbalance": "smote"},
}

# (name, model A, model B): "is A better than B?"
COMPARISONS = [
    ("H1_stage1_forecast_helps", "lstm_tyre_aware", "lstm_no_stage1"),
    ("H2_sequence_beats_trees", "lstm_tyre_aware", "xgboost"),
    ("main_model_vs_published_bilstm", "lstm_tyre_aware", "bilstm_replica"),
]


@dataclass
class Stage2Settings:
    window_laps: int = 10
    inner_valid_share: float = 0.2
    seeds: list[int] = field(default_factory=lambda: [42])
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    lstm_hidden: int = 64
    dropout: float = 0.2
    bilstm_units: list[int] = field(default_factory=lambda: [256, 128, 64])
    bilstm_dropout: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.3])
    bilstm_batch_size: int = 32
    bilstm_learning_rate: float = 5e-4
    bilstm_positive_weight: float = 3.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pit_window_laps: int = 2
    bootstrap_samples: int = 1000
    experiments: list[str] = field(default_factory=lambda: list(EXPERIMENTS))
    mlflow: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> Stage2Settings:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown stage2 settings: {sorted(unknown)}")
        settings = cls(**raw)
        bad = [e for e in settings.experiments if e not in EXPERIMENTS]
        if bad:
            raise ValueError(f"Unknown stage2 experiments: {bad}. Choose from {list(EXPERIMENTS)}")
        if len(settings.bilstm_units) != len(settings.bilstm_dropout):
            raise ValueError("bilstm_units and bilstm_dropout must have the same length")
        if not settings.seeds:
            raise ValueError("stage2.seeds must list at least one seed")
        return settings


# --------------------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------------------
def feature_columns(stage1: bool) -> list[str]:
    return BASE_FEATURES + (STAGE1_FEATURES if stage1 else [])


def split_masks(df: pd.DataFrame, settings: Stage2Settings) -> dict[str, np.ndarray]:
    """Fit / inner-validation / validation masks. The inner split is by race and identical for all models."""
    trainable = ((df["Split"] == "train") & ~df["RuleForced"].astype(bool)).to_numpy()
    valid = df["Split"].isin(["valid", "validation"]).to_numpy()
    races = df.loc[trainable, "RaceID"].to_numpy()
    if len(np.unique(races)) < 2:
        raise ValueError("Stage 2 needs at least two training races.")
    if not valid.any():
        raise ValueError("Stage 2 needs validation-season laps.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=settings.inner_valid_share, random_state=settings.seeds[0]
    )
    fit_pos, inner_pos = next(splitter.split(races, groups=races))
    trainable_idx = np.flatnonzero(trainable)
    fit, inner = np.zeros(len(df), bool), np.zeros(len(df), bool)
    fit[trainable_idx[fit_pos]] = True
    inner[trainable_idx[inner_pos]] = True
    return {"fit": fit, "inner": inner, "valid": valid}


class Scaler:
    """Median fill then standardise, with statistics from the fitting laps only."""

    def fit(self, df: pd.DataFrame, cols: list[str]) -> Scaler:
        data = df[cols].astype(float)
        self.cols = cols
        self.median = data.median().fillna(0.0)
        filled = data.fillna(self.median)
        self.mean = filled.mean()
        self.std = filled.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        data = df[self.cols].astype(float).fillna(self.median)
        return ((data - self.mean) / self.std).to_numpy(np.float32)

    def to_dict(self) -> dict:
        return {
            "cols": list(self.cols),
            "median": {k: float(v) for k, v in self.median.items()},
            "mean": {k: float(v) for k, v in self.mean.items()},
            "std": {k: float(v) for k, v in self.std.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Scaler:
        scaler = cls()
        scaler.cols = list(raw["cols"])
        scaler.median = pd.Series(raw["median"], dtype=float)[scaler.cols]
        scaler.mean = pd.Series(raw["mean"], dtype=float)[scaler.cols]
        scaler.std = pd.Series(raw["std"], dtype=float)[scaler.cols]
        return scaler


def make_windows(df: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    """Windows of the last `window` laps of the same driver in the same race, ending at each lap.

    Shape (laps, window, features + 1). The extra channel is 1 for a real lap and 0 for left padding
    (the first laps of a race have fewer than `window` laps of history). `df` must be sorted by
    race, driver and lap, which `prepare` guarantees.
    """
    n, n_feat = values.shape
    key = df["RaceID"].astype(str).to_numpy() + "|" + df["Driver"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    idx = np.arange(n)
    row_start = starts[np.searchsorted(starts, idx, side="right") - 1]
    out = np.zeros((n, window, n_feat + 1), np.float32)
    for j in range(window):
        src = idx - (window - 1 - j)
        ok = src >= row_start
        out[ok, j, :n_feat] = values[src[ok]]
        out[ok, j, n_feat] = 1.0
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_FEATURES + STAGE1_FEATURES + ["PitLap", "Split", "RuleForced"] if c not in df]
    if missing:
        raise ValueError(f"Stage 2 input is missing columns {missing}. Run Stage 1 first.")
    out = df.sort_values(ORDER).reset_index(drop=True)
    out["Censored"] = out["Censored"].astype(float)
    return out


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------
def typical_stint_lengths(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Median tyre age at a pit stop, per circuit and compound and per compound, from training races."""
    stops = train[train["PitLap"] == 1]
    return (
        stops.groupby(["EventName", "Compound"])["TyreLife"].median(),
        stops.groupby("Compound")["TyreLife"].median(),
    )


def rule_based_scores(
    df: pd.DataFrame, by_event: pd.Series, by_compound: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Pit on the first lap on which tyre age reaches the typical stop age.

    The typical age is a median and can end in .5; it is rounded up so there is always exactly one
    lap per stint on which the rule fires. Score = closeness of tyre age to that lap.
    """
    keys = pd.MultiIndex.from_arrays([df["EventName"], df["Compound"]])
    typical = pd.Series(by_event.reindex(keys).to_numpy(), index=df.index)
    typical = typical.fillna(df["Compound"].map(by_compound)).fillna(by_compound.median())
    typical = np.ceil(typical - 1e-9)
    distance = (df["TyreLife"] - typical).abs().to_numpy(float)
    return 1.0 / (1.0 + distance), (distance < 0.5).astype(int)


def _torch():
    import torch
    from torch import nn

    return torch, nn


def build_lstm(n_inputs: int, s: Stage2Settings):
    torch, nn = _torch()

    class PitLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_inputs, s.lstm_hidden, batch_first=True)
            self.drop = nn.Dropout(s.dropout)
            self.head = nn.Linear(s.lstm_hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    return PitLSTM()


def build_bilstm(n_inputs: int, s: Stage2Settings):
    """Three stacked bidirectional layers, as published. PyTorch has no recurrent dropout, so dropout is
    applied between layers only (a documented difference from the Keras original)."""
    torch, nn = _torch()

    class BiLSTMReplica(nn.Module):
        def __init__(self):
            super().__init__()
            sizes = [n_inputs] + [2 * u for u in s.bilstm_units[:-1]]
            self.layers = nn.ModuleList(
                nn.LSTM(i, u, batch_first=True, bidirectional=True)
                for i, u in zip(sizes, s.bilstm_units, strict=True)
            )
            self.drops = nn.ModuleList(nn.Dropout(p) for p in s.bilstm_dropout)
            self.head = nn.Linear(2 * s.bilstm_units[-1], 1)

        def forward(self, x):
            for lstm, drop in zip(self.layers, self.drops, strict=True):
                x, _ = lstm(x)
                x = drop(x)
            return self.head(x[:, -1, :]).squeeze(-1)

    return BiLSTMReplica()


def make_loss(imbalance: str, y_fit: np.ndarray, s: Stage2Settings):
    torch, nn = _torch()
    pos = max(float(y_fit.sum()), 1.0)
    if imbalance == "class_weight":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_fit) - pos) / pos))
    if imbalance == "smote_and_weight":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(s.bilstm_positive_weight))
    if imbalance == "smote":
        return nn.BCEWithLogitsLoss()
    if imbalance == "focal":

        def focal(logits, target):
            p = torch.sigmoid(logits)
            ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p_t = p * target + (1 - p) * (1 - target)
            alpha_t = s.focal_alpha * target + (1 - s.focal_alpha) * (1 - target)
            return (alpha_t * (1 - p_t) ** s.focal_gamma * ce).mean()

        return focal
    raise ValueError(f"unknown imbalance method {imbalance}")


def smote_windows(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE on whole windows (flattened), training data only. Synthetic windows interpolate entire
    lap sequences between neighbouring real pit windows."""
    from imblearn.over_sampling import SMOTE

    n, w, f = X.shape
    k = int(min(5, max(1, y.sum() - 1)))
    Xs, ys = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X.reshape(n, -1), y)
    return Xs.reshape(-1, w, f).astype(np.float32), ys.astype(np.float32)


def predict_torch(model, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    torch, _ = _torch()
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i : i + batch]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_torch(model, loss_fn, X_fit, y_fit, X_inner, y_inner, lr, batch_size, s: Stage2Settings, seed: int):
    """Adam, early stopping on held-out PR-AUC; returns the best model and its training history."""
    torch, nn = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if isinstance(loss_fn, nn.Module):  # moves the class-weight tensor to the same device as the model
        loss_fn.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    Xf, yf = torch.from_numpy(X_fit), torch.from_numpy(y_fit.astype(np.float32))
    best_ap, best_state, wait, history = -1.0, copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(s.max_epochs):
        model.train()
        order = torch.randperm(len(Xf), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch_size):
            b = order[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xf[b].to(device)), yf[b].to(device))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        ap = float(average_precision_score(y_inner, predict_torch(model, X_inner)))
        history.append({"epoch": epoch + 1, "loss": total / len(order), "inner_pr_auc": ap})
        if ap > best_ap:
            best_ap, best_state, wait = ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= s.patience:
                break
    model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        torch, _ = _torch()
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_experiment(
    name: str, df: pd.DataFrame, masks: dict, s: Stage2Settings, seed: int, model_dir: Path | None = None
) -> dict:
    """Train one experiment with one seed. Returns inner and validation scores plus the chosen threshold."""
    spec = EXPERIMENTS[name]
    fit, inner, valid = masks["fit"], masks["inner"], masks["valid"]
    y = df["PitLap"].to_numpy(int)
    _set_seed(seed)
    info: dict = {"seed": seed}

    if spec["kind"] == "rule":
        by_event, by_comp = typical_stint_lengths(df[fit | inner])
        score, pred = rule_based_scores(df, by_event, by_comp)
        return {
            **info,
            "valid_score": score[valid],
            "valid_pred": pred[valid],
            "threshold": None,
            "is_probability": False,
            "rule_tables": {
                "by_event": [[event, comp, float(v)] for (event, comp), v in by_event.items()],
                "by_compound": {comp: float(v) for comp, v in by_comp.items()},
            },
        }

    cols = feature_columns(spec["stage1"])
    if spec["kind"] == "xgb":
        X = df[cols].astype(float)
        pos = y[fit].sum()
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(fit.sum() - pos) / max(pos, 1),
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[fit], y[fit])
        inner_score = model.predict_proba(X[inner])[:, 1]
        valid_score = model.predict_proba(X[valid])[:, 1]
        if model_dir is not None:
            model.save_model(model_dir / f"{name}_seed{seed}.json")
        info["feature_importance"] = dict(zip(cols, map(float, model.feature_importances_), strict=True))
    else:
        scaler = Scaler().fit(df[fit], cols)
        windows = make_windows(df, scaler.transform(df), s.window_laps)
        X_fit, y_fit = windows[fit], y[fit]
        if spec["imbalance"] in ("smote", "smote_and_weight"):
            X_fit, y_fit = smote_windows(X_fit, y_fit, seed)
        n_inputs = windows.shape[2]
        if spec["kind"] == "bilstm":
            model, lr, batch = build_bilstm(n_inputs, s), s.bilstm_learning_rate, s.bilstm_batch_size
        else:
            model, lr, batch = build_lstm(n_inputs, s), s.learning_rate, s.batch_size
        loss_fn = make_loss(spec["imbalance"], y_fit, s)
        model, history = train_torch(
            model, loss_fn, X_fit, y_fit, windows[inner], y[inner], lr, batch, s, seed
        )
        inner_score, valid_score = predict_torch(model, windows[inner]), predict_torch(model, windows[valid])
        info["history"] = history
        if model_dir is not None:
            torch, _ = _torch()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "experiment": name,
                    "features": cols,
                    "scaler": scaler.to_dict(),
                    "settings": asdict(s),
                },
                model_dir / f"{name}_seed{seed}.pt",
            )

    threshold = best_f1_threshold(y[inner], inner_score)
    return {
        **info,
        "inner_score": inner_score,
        "valid_score": valid_score,
        "valid_pred": (valid_score >= threshold).astype(int),
        "threshold": threshold,
        "is_probability": True,
    }


def _summarise(values: list[dict]) -> dict:
    keys = [k for k, v in values[0].items() if isinstance(v, (int, float)) and v is not None]
    return {
        k: {"mean": float(np.mean([v[k] for v in values])), "std": float(np.std([v[k] for v in values]))}
        for k in keys
    }


def _mlflow_log(enabled: bool, run_name: str, params: dict, metrics: dict) -> None:
    if not enabled:
        return
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("stage2.mlflow is true but MLflow is not installed: pip install mlflow") from exc
    mlflow.set_experiment("f1pit-stage2")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})


def run_stage2(
    feats: pd.DataFrame, cfg: Config, settings: Stage2Settings | None = None, model_dir: Path | None = None
) -> tuple[dict, pd.DataFrame]:
    s = settings or Stage2Settings.from_dict(cfg.stage2)
    df = prepare(feats)
    masks = split_masks(df, s)
    y = df["PitLap"].to_numpy(int)
    valid_meta = df.loc[masks["valid"], ["RaceID", "EventName", "Driver", "LapNumber", "PitLap"]].reset_index(
        drop=True
    )
    y_valid, y_inner = y[masks["valid"]], y[masks["inner"]]
    race_ids = valid_meta["RaceID"].to_numpy()
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "protocol": {
            "fit_races": int(df.loc[masks["fit"], "RaceID"].nunique()),
            "inner_validation_races": int(df.loc[masks["inner"], "RaceID"].nunique()),
            "validation_races": int(df.loc[masks["valid"], "RaceID"].nunique()),
            "window_laps": s.window_laps,
            "seeds": s.seeds,
            "threshold_rule": "maximise F1 on the inner-validation (held-out training) races",
            "headline_metric": "pr_auc on the validation season",
        },
        "experiments": {},
    }
    manifest: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": asdict(s),
        "train_seasons": list(cfg.train_seasons),
        "valid_seasons": list(cfg.valid_seasons),
        "quick": bool(cfg.quick),
        "fit_races": sorted(df.loc[masks["fit"], "RaceID"].unique().tolist()),
        "inner_validation_races": sorted(df.loc[masks["inner"], "RaceID"].unique().tolist()),
        "threshold_rule": results["protocol"]["threshold_rule"],
        "experiments": {},
    }
    predictions = valid_meta.copy()
    ensemble: dict[str, np.ndarray] = {}
    ensemble_pred: dict[str, np.ndarray] = {}

    for name in s.experiments:
        spec = EXPERIMENTS[name]
        seeds = s.seeds if spec["kind"] in ("lstm", "bilstm", "xgb") else [s.seeds[0]]
        runs, per_seed = [], []
        for seed in seeds:
            run = run_experiment(name, df, masks, s, seed, model_dir)
            m = classification_metrics(y_valid, run["valid_score"], run["valid_pred"], run["is_probability"])
            m["pit_within_window"] = pit_window_hit_rate(valid_meta, run["valid_pred"], s.pit_window_laps)
            m["threshold"] = run["threshold"]
            per_seed.append(m)
            runs.append(run)
            _mlflow_log(s.mlflow, f"{name}_seed{seed}", {"experiment": name, "seed": seed, **spec}, m)

        score = np.mean([r["valid_score"] for r in runs], axis=0)
        if runs[0]["is_probability"]:
            inner_mean = np.mean([r["inner_score"] for r in runs], axis=0)
            threshold = best_f1_threshold(y_inner, inner_mean)
            pred = (score >= threshold).astype(int)
        else:
            threshold, pred = None, runs[0]["valid_pred"]
        combined = classification_metrics(y_valid, score, pred, runs[0]["is_probability"])
        combined["pit_within_window"] = pit_window_hit_rate(valid_meta, pred, s.pit_window_laps)
        combined["threshold"] = threshold
        combined["pr_auc_ci"] = race_bootstrap(
            race_ids, y_valid, score, n_samples=s.bootstrap_samples, seed=s.seeds[0]
        )
        entry = {
            "spec": spec,
            "seed_ensemble": combined,
            "per_seed_mean_std": _summarise(per_seed),
            "per_seed": per_seed,
        }
        for key in ("history", "feature_importance"):
            if key in runs[0]:
                entry[key] = [r[key] for r in runs]
        results["experiments"][name] = entry
        extension = {"xgb": "json", "lstm": "pt", "bilstm": "pt"}.get(spec["kind"])
        manifest["experiments"][name] = {
            "kind": spec["kind"],
            "stage1": spec["stage1"],
            "is_probability": bool(runs[0]["is_probability"]),
            "features": None if spec["kind"] == "rule" else feature_columns(spec["stage1"]),
            "seeds": list(seeds),
            "files": [] if extension is None else [f"{name}_seed{seed}.{extension}" for seed in seeds],
            "per_seed_threshold": [r["threshold"] for r in runs],
            "threshold": threshold,
            **({"rule_tables": runs[0]["rule_tables"]} if "rule_tables" in runs[0] else {}),
        }
        ensemble[name], ensemble_pred[name] = score, pred
        predictions[f"{name}_score"], predictions[f"{name}_pred"] = score, pred

    results["comparisons"] = {}
    for label, a, b in COMPARISONS:
        if a in ensemble and b in ensemble:
            results["comparisons"][label] = {
                "model_a": a,
                "model_b": b,
                "pr_auc_difference": race_bootstrap(
                    race_ids,
                    y_valid,
                    ensemble[a],
                    ensemble[b],
                    n_samples=s.bootstrap_samples,
                    seed=s.seeds[0],
                ),
                "mcnemar": mcnemar_test(y_valid, ensemble_pred[a], ensemble_pred[b]),
            }
    if model_dir is not None:
        write_manifest(manifest, model_dir)
    return results, predictions


def write_manifest(manifest: dict, model_dir: Path) -> dict:
    """Save what Phase 6 needs to score new races without refitting or retuning anything."""
    manifest = json.loads(json.dumps(_clean_json(manifest), default=_json_default, allow_nan=False))
    body = json.dumps(manifest, sort_keys=True)
    manifest = {**manifest, "fingerprint": hashlib.sha256(body.encode()).hexdigest()[:16]}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _clean_json(o):
    """Replace NaN and infinity (not valid JSON) with null, recursively."""
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def print_summary(results: dict) -> None:
    print(
        f"Validation season: {results['protocol']['validation_races']} races "
        f"(models fitted on {results['protocol']['fit_races']} training races, "
        f"{results['protocol']['inner_validation_races']} held out for early stopping and threshold)"
    )
    print(f"{'experiment':24s} {'PR-AUC (95% CI)':24s} {'F1':>6s} {'recall':>7s} {'±2 laps':>8s}")
    for name, e in results["experiments"].items():
        m = e["seed_ensemble"]
        ci = m["pr_auc_ci"]
        print(
            f"{name:24s} {m['pr_auc']:.3f} ({ci['ci_low']:.3f}-{ci['ci_high']:.3f})    "
            f"{m['f1']:6.3f} {m['recall']:7.3f} {m['pit_within_window']:8.3f}"
        )
    for label, c in results["comparisons"].items():
        d = c["pr_auc_difference"]
        print(
            f"{label}: PR-AUC difference {d['estimate']:+.3f} "
            f"(95% CI {d['ci_low']:+.3f} to {d['ci_high']:+.3f}), "
            f"bootstrap p {_p_text(d['p_not_better'])}, McNemar p {_p_text(c['mcnemar']['p_value'])}"
        )


def _p_text(p: float) -> str:
    """'= 0.031' or '< 0.001': a p-value is never printed as 0.000."""
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", nargs="*", help="run only these experiments")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw = dict(cfg.stage2)
    if args.experiments:
        raw["experiments"] = args.experiments
    settings = Stage2Settings.from_dict(raw)
    feats = pd.read_parquet(cfg.processed_dir / "features_stage1.parquet")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = run_stage2(feats, cfg, settings, model_dir=cfg.results_dir / "models")
    (cfg.results_dir / "stage2_metrics.json").write_text(
        json.dumps(_clean_json(results), indent=2, default=_json_default, allow_nan=False)
    )
    predictions.to_parquet(cfg.results_dir / "stage2_validation_predictions.parquet", index=False)
    print_summary(results)
    if args.experiments:
        print(
            "NOTE: only the experiments named above are now in results/models/manifest.json and "
            "stage2_metrics.json. The final evaluation covers exactly those, so run every experiment "
            "you need (including both models of each hypothesis) in ONE Stage 2 run before evaluating."
        )


if __name__ == "__main__":
    main()
