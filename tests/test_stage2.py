import numpy as np
import pandas as pd
import pytest

from f1pit.features import build_features, fit_fuel_slopes
from f1pit.metrics import best_f1_threshold, mcnemar_test, race_bootstrap
from f1pit.stage1 import run_stage1
from f1pit.stage2 import (
    Scaler,
    Stage2Settings,
    make_loss,
    make_windows,
    prepare,
    rule_based_scores,
    run_experiment,
    run_stage2,
    smote_windows,
    split_masks,
)

torch = pytest.importorskip("torch")

TINY = dict(
    window_laps=5,
    seeds=[1],
    max_epochs=3,
    patience=2,
    batch_size=256,
    lstm_hidden=8,
    bilstm_units=[8, 4],
    bilstm_dropout=[0.1, 0.1],
    bilstm_batch_size=256,
    bootstrap_samples=50,
)


@pytest.fixture
def stage1_out(laps, cfg):
    slopes = fit_fuel_slopes(laps[(laps["Split"] == "train") & ~laps["RuleForced"]])
    out, _ = run_stage1(build_features(laps, slopes), cfg, model_names=("linear",))
    return prepare(out)


def tiny_frame():
    return pd.DataFrame(
        {"RaceID": ["r"] * 5, "Driver": ["A", "A", "A", "B", "B"], "LapNumber": [1, 2, 3, 1, 2]}
    )


def test_windows_are_causal_padded_and_stay_with_one_driver():
    df = tiny_frame()
    values = np.arange(5, dtype=np.float32).reshape(-1, 1) + 1  # A: 1, 2, 3   B: 4, 5
    w = make_windows(df, values, window=3)
    assert w[2, :, 0].tolist() == [1, 2, 3] and w[2, :, 1].tolist() == [1, 1, 1]
    assert w[0, :, 0].tolist() == [0, 0, 1] and w[0, :, 1].tolist() == [0, 0, 1]  # left padding
    assert w[4, :, 0].tolist() == [0, 4, 5]  # never sees driver A
    # a window ending at lap L does not change when later laps are removed
    assert np.array_equal(make_windows(df.iloc[:2], values[:2], 3), w[:2])


def test_split_masks_keep_races_apart(stage1_out):
    s = Stage2Settings.from_dict(TINY)
    m = split_masks(stage1_out, s)
    fit_races = set(stage1_out.loc[m["fit"], "RaceID"])
    inner_races = set(stage1_out.loc[m["inner"], "RaceID"])
    assert fit_races and inner_races and fit_races.isdisjoint(inner_races)
    assert "2023_04" not in fit_races | inner_races  # rule-forced race excluded
    assert set(stage1_out.loc[m["valid"], "Split"]) == {"valid"}
    assert not (m["valid"] & (m["fit"] | m["inner"])).any()


def test_scaler_statistics_come_from_fitting_rows_only(stage1_out):
    m = split_masks(stage1_out, Stage2Settings.from_dict(TINY))
    scaler = Scaler().fit(stage1_out[m["fit"]], ["TyreLife"])
    assert scaler.median["TyreLife"] == stage1_out.loc[m["fit"], "TyreLife"].median()


def test_rule_based_scores():
    df = pd.DataFrame(
        {"EventName": ["X", "X", "Y"], "Compound": ["SOFT", "SOFT", "HARD"], "TyreLife": [18.0, 12.0, 30.0]}
    )
    by_event = pd.Series({("X", "SOFT"): 18.0})
    by_event.index = pd.MultiIndex.from_tuples(by_event.index)
    score, pred = rule_based_scores(df, by_event, pd.Series({"SOFT": 20.0, "HARD": 30.0}))
    assert pred.tolist() == [1, 0, 1]  # Y/HARD falls back to the compound median (30)
    assert score[0] == 1.0 and score[1] == pytest.approx(1 / 7)


def test_rule_fires_once_when_the_typical_stop_age_is_a_half():
    df = pd.DataFrame(
        {"EventName": ["X"] * 4, "Compound": ["SOFT"] * 4, "TyreLife": [16.0, 17.0, 18.0, 19.0]}
    )
    by_event = pd.Series({("X", "SOFT"): 17.5})  # the median of an even number of stops
    by_event.index = pd.MultiIndex.from_tuples(by_event.index)
    score, pred = rule_based_scores(df, by_event, pd.Series({"SOFT": 20.0}))
    assert pred.tolist() == [0, 0, 1, 0] and score.argmax() == 2


def test_smote_balances_training_windows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5, 3)).astype(np.float32)
    y = np.r_[np.ones(20), np.zeros(180)].astype(int)
    Xs, ys = smote_windows(X, y, seed=0)
    assert Xs.shape[1:] == (5, 3) and ys.sum() == (ys == 0).sum() == 180


def test_focal_loss_matches_half_bce_without_focusing():
    s = Stage2Settings.from_dict({**TINY, "focal_alpha": 0.5, "focal_gamma": 0.0})
    logits, target = torch.tensor([2.0, -1.0, 0.5]), torch.tensor([1.0, 0.0, 1.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert float(make_loss("focal", np.array([1, 0, 1]), s)(logits, target)) == pytest.approx(
        0.5 * float(bce)
    )


def test_threshold_is_chosen_on_inner_validation_races(stage1_out):
    s = Stage2Settings.from_dict(TINY)
    m = split_masks(stage1_out, s)
    run = run_experiment("lstm_tyre_aware", stage1_out, m, s, seed=1)
    y_inner = stage1_out.loc[m["inner"], "PitLap"].to_numpy()
    assert run["threshold"] == pytest.approx(best_f1_threshold(y_inner, run["inner_score"]))
    assert len(run["valid_score"]) == m["valid"].sum()


def test_bootstrap_and_mcnemar_basics():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    races = np.repeat(["a", "b", "c", "d"], 3)
    perfect, noise = y.astype(float), np.linspace(0, 1, 12)
    d = race_bootstrap(races, y, perfect, noise, n_samples=200, seed=0)
    assert d["estimate"] > 0 and d["p_not_better"] < 0.05
    t = mcnemar_test(y, y, 1 - y)
    assert t["only_a_correct"] == 12 and t["p_value"] < 0.01


def test_settings_are_checked():
    with pytest.raises(ValueError, match="Unknown stage2 settings"):
        Stage2Settings.from_dict({"windw_laps": 5})
    with pytest.raises(ValueError, match="Unknown stage2 experiments"):
        Stage2Settings.from_dict({"experiments": ["lstm_magic"]})


def test_run_stage2_end_to_end(stage1_out, cfg, tmp_path):
    s = Stage2Settings.from_dict(TINY)
    results, preds = run_stage2(stage1_out, cfg, s, model_dir=tmp_path / "models")
    assert set(results["experiments"]) == set(s.experiments)
    assert len(preds) == (stage1_out["Split"] == "valid").sum()
    for name, e in results["experiments"].items():
        m = e["seed_ensemble"]
        assert 0 <= m["pr_auc"] <= 1 and 0 <= m["recall"] <= 1, name
    no_skill = results["experiments"]["xgboost"]["seed_ensemble"]["no_skill_pr_auc"]
    assert results["experiments"]["xgboost"]["seed_ensemble"]["pr_auc"] > 3 * no_skill
    assert results["experiments"]["lstm_tyre_aware"]["seed_ensemble"]["pr_auc"] > no_skill
    assert set(results["comparisons"]) == {
        "H1_stage1_forecast_helps",
        "H2_sequence_beats_trees",
        "main_model_vs_published_bilstm",
    }
    saved = {p.name for p in (tmp_path / "models").iterdir()}
    assert "lstm_tyre_aware_seed1.pt" in saved and "xgboost_seed1.json" in saved


def test_results_json_has_no_nan():
    import json

    from f1pit.stage2 import _clean_json

    cleaned = _clean_json({"a": float("nan"), "b": [1.0, np.float64("inf")], "c": {"d": 0.5}})
    assert cleaned == {"a": None, "b": [1.0, None], "c": {"d": 0.5}}
    json.dumps(cleaned, allow_nan=False)  # raises if any NaN were left


def test_mlflow_logging(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from f1pit.stage2 import _mlflow_log

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    _mlflow_log(
        True,
        "lstm_tyre_aware_seed1",
        {"experiment": "lstm_tyre_aware", "seed": 1},
        {"pr_auc": 0.5, "brier": None},
    )
    runs = mlflow.search_runs(experiment_names=["f1pit-stage2"])
    assert len(runs) == 1 and runs.loc[0, "metrics.pr_auc"] == 0.5
    _mlflow_log(False, "skipped", {}, {})  # disabled: does nothing
    assert len(mlflow.search_runs(experiment_names=["f1pit-stage2"])) == 1
