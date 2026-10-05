import numpy as np
import pandas as pd
import pytest
from synthetic import DEG_RATE

from f1pit.features import build_features, fit_fuel_slopes
from f1pit.stage1 import (
    LinearDegradation,
    XGBDegradation,
    actual_laps_to_threshold,
    add_stint_state,
    laps_to_threshold,
    out_of_fold,
    projection,
    run_stage1,
)


@pytest.fixture
def feats(laps):
    slopes = fit_fuel_slopes(laps[(laps["Split"] == "train") & ~laps["RuleForced"]])
    return add_stint_state(build_features(laps, slopes))


@pytest.fixture
def train(feats):
    return feats[(feats["Split"] == "train") & ~feats["RuleForced"]]


def test_linear_model_recovers_true_degradation_rates(train):
    model = LinearDegradation().fit(train)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        assert model.by_compound[comp][1] == pytest.approx(DEG_RATE[comp], abs=0.015)


def test_laps_to_threshold_maths():
    current = np.array([0.2, 1.2, 0.0])
    rates = np.array([0.1, 0.1, 0.0])
    projected = current[:, None] + rates[:, None] * np.arange(1, 41)
    laps, censored = laps_to_threshold(current, projected, threshold=0.95)
    assert laps.tolist() == [8, 0, 40]  # 0.2 + 0.1 x 8 = 1.0 is the first value >= 0.95
    assert censored.tolist() == [False, False, True]


def test_projection_uses_no_future_laps(feats, train):
    """LapsToThreshold for laps 1..L is identical whether or not later laps exist."""
    model = LinearDegradation().fit(train)
    race = feats[feats["RaceID"] == "2024_01"].drop(columns=["LapsInStint", "SmoothedDeg", "ObservedDeg"])
    full = add_stint_state(race)
    cur_f, proj_f = projection(model, full, 40)
    full = full.assign(Pred=laps_to_threshold(cur_f, proj_f, 1.0)[0], Cur=cur_f)
    for cut in [8, 25, 44]:
        part = add_stint_state(race[race["LapNumber"] <= cut])
        cur_p, proj_p = projection(model, part, 40)
        part = part.assign(Pred=laps_to_threshold(cur_p, proj_p, 1.0)[0], Cur=cur_p)
        merged = part.merge(full, on=["Driver", "LapNumber"], suffixes=("_part", "_full"))
        assert len(merged) == len(part)
        assert (merged["Pred_part"] == merged["Pred_full"]).all()
        assert np.allclose(merged["Cur_part"], merged["Cur_full"])


class SpyModel(LinearDegradation):
    instances: list = []

    def fit(self, df):
        self.fit_races = set(df["RaceID"])
        self.predicted_races: set = set()
        SpyModel.instances.append(self)
        return super().fit(df)

    def predict(self, df):
        self.predicted_races |= set(df["RaceID"])
        return super().predict(df)


def test_out_of_fold_never_predicts_a_race_it_was_fitted_on(train, cfg):
    SpyModel.instances = []
    oof = out_of_fold(SpyModel, train, cfg)
    assert len(SpyModel.instances) == min(cfg.oof_folds, train["RaceID"].nunique())
    for spy in SpyModel.instances:
        assert spy.fit_races.isdisjoint(spy.predicted_races)
    assert oof.index.equals(train.index) and oof["LapsToThreshold"].notna().all()


def test_xgboost_degradation_never_falls_as_the_stint_ages(train):
    model = XGBDegradation().fit(train)
    rows = train.sample(200, random_state=0)
    preds = []
    for k in range(1, 31):
        state = rows.assign(LapsInStint=k, TyreLife=rows["TyreLife"] - rows["LapsInStint"] + k)
        preds.append(model.predict(state))
    assert (np.diff(np.column_stack(preds), axis=1) >= -1e-9).all()


def test_actual_laps_to_threshold_on_a_tiny_stint():
    df = pd.DataFrame(
        {
            "RaceID": "r",
            "Driver": "A",
            "Stint": 1.0,
            "LapNumber": [1.0, 2, 3, 4, 5],
            "SmoothedDeg": [0.0, 0.3, 0.7, 1.1, 1.4],
        }
    )
    # threshold 1.0 is first reached on lap 4; from the end of lap 1 that is 3 laps away
    assert actual_laps_to_threshold(df, 1.0).tolist()[:4] == [4.0, 3.0, 2.0, 1.0]
    assert np.isnan(actual_laps_to_threshold(df, 1.0)[4])  # already past it
    assert np.isnan(actual_laps_to_threshold(df, 2.0)).all()  # never reached: censored


def test_run_stage1_end_to_end(feats, cfg):
    df, metrics = run_stage1(feats.drop(columns=["LapsInStint", "SmoothedDeg", "ObservedDeg"]), cfg)
    assert metrics["selected_model"] in {"linear", "xgboost"}
    for name in ["linear", "xgboost"]:
        m = metrics[name]
        assert m["mae_5lap_s"] < m["persistence_mae_5lap_s"]  # better than assuming no further wear
        primary = next(t for t in m["thresholds"] if t["threshold_s"] == cfg.primary_threshold_s)
        assert primary["mae_laps"] < 4
    assert df["LapsToThreshold"].between(0, cfg.horizon_laps).all()
    fitted = (df["Split"] == "train") & ~df["RuleForced"]
    assert (df.loc[fitted, "LapsToThresholdSource"] == "out_of_fold").all()
    assert (df.loc[~fitted, "LapsToThresholdSource"] == "fitted_on_train").all()
    assert len(df) == len(feats)


def test_selection_keeps_simple_model_unless_clearly_better():
    from f1pit.stage1 import select_model

    names, key = ("linear", "xgboost"), "mae_5lap_s"
    tie = {"linear": {key: 0.2320}, "xgboost": {key: 0.2318}}
    clear = {"linear": {key: 0.2320}, "xgboost": {key: 0.2000}}
    worse = {"linear": {key: 0.2320}, "xgboost": {key: 0.2500}}
    assert select_model(tie, names, key, 0.01) == "linear"
    assert select_model(clear, names, key, 0.01) == "xgboost"
    assert select_model(worse, names, key, 0.01) == "linear"
