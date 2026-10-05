import numpy as np
import pandas as pd
import pytest
from synthetic import TRUE_FUEL_SLOPE

from f1pit.features import FEATURES, add_car_ahead_features, build_features, fit_fuel_slopes, race_fuel_slope


@pytest.fixture
def slopes(laps):
    return fit_fuel_slopes(laps[(laps["Split"] == "train") & ~laps["RuleForced"]])


def test_fuel_slope_recovers_the_true_effect(laps):
    race = laps[laps["RaceID"] == "2022_01"]
    assert race_fuel_slope(race) == pytest.approx(TRUE_FUEL_SLOPE, abs=0.01)


def test_fuel_slopes_come_from_training_races_only(slopes):
    assert all(rid.startswith(("2022", "2023")) for rid in slopes["per_race"])
    assert "2023_04" not in slopes["per_race"]  # rule-forced race excluded


def test_validation_races_use_training_slopes(laps, slopes):
    feats = build_features(laps, slopes)
    valid = feats[feats["Split"] == "valid"]
    expected = valid["EventName"].map(slopes["by_event"]).fillna(slopes["global"])
    assert np.allclose(valid["FuelSlope"], expected)


def test_no_feature_uses_future_laps(laps, slopes):
    """Features for laps 1..L must be identical whether or not later laps exist."""
    race = laps[laps["RaceID"] == "2022_01"]
    full = build_features(race, slopes).set_index(["Driver", "LapNumber"])
    for cut in [5, 19, 30, 50]:
        part = build_features(race[race["LapNumber"] <= cut], slopes).set_index(["Driver", "LapNumber"])
        cols = FEATURES + ["Degradation_s", "StintBase"]
        pd.testing.assert_frame_equal(part[cols], full.loc[part.index, cols], check_dtype=False)


def test_degradation_grows_with_tyre_age(laps, slopes):
    feats = build_features(laps, slopes)
    clean = feats[feats["CleanLap"] & (feats["Compound"] == "HARD")]
    young = clean.loc[clean["TyreLife"].between(4, 6), "Degradation_s"].median()
    old = clean.loc[clean["TyreLife"].between(25, 30), "Degradation_s"].median()
    assert old > young + 0.5  # 0.04 s/lap over ~22 laps is about 0.9 s


def test_car_ahead_features_on_a_tiny_example():
    df = pd.DataFrame(
        {
            "RaceID": ["r"] * 4,
            "Driver": ["A", "B", "A", "B"],
            "LapNumber": [1, 1, 2, 2],
            "Position": [1.0, 2.0, 2.0, 1.0],
            "Time_s": [100.0, 101.5, 210.0, 205.0],
            "PitLap": [1, 0, 0, 0],
            "PrevPosition": [np.nan, np.nan, 1.0, 2.0],
            "PrevTime_s": [np.nan, np.nan, 100.0, 101.5],
        }
    )
    out = add_car_ahead_features(df).set_index(["Driver", "LapNumber"])
    assert len(out) == 4
    assert out.loc[("B", 2), "GapAhead_s"] == pytest.approx(1.5)  # A was 1.5 s ahead after lap 1
    assert out.loc[("B", 2), "AheadPitted"] == 1  # and A pitted on lap 1
    assert out.loc[("A", 2), "AheadPitted"] == 0  # A was leading: nobody ahead
