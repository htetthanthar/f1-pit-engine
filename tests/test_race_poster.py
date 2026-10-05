import numpy as np
import pandas as pd
import pytest

from f1pit import race_poster as rp


def tiny_race() -> pd.DataFrame:
    """Three drivers, eight laps. A wins and stops on lap 4; C retires after lap 5; lap 6 is a safety car."""
    rows = []
    for driver, pace, last in (("A", 90.0, 8), ("B", 90.5, 8), ("C", 91.0, 5)):
        time = 0.0
        for lap in range(1, last + 1):
            pit = int(driver == "A" and lap == 4)
            lap_time = pace + 20.0 * pit
            time += lap_time
            second_stint = driver == "A" and lap > 4
            rows.append(
                {
                    "RaceID": "2025_01",
                    "Year": 2025,
                    "EventName": "Test Grand Prix",
                    "Split": "test",
                    "RuleForced": False,
                    "Driver": driver,
                    "Team": "T" + driver,
                    "LapNumber": float(lap),
                    "Stint": 2.0 if second_stint else 1.0,
                    "Compound": "HARD" if second_stint else "SOFT",
                    "TyreLife": float(lap - 4 if second_stint else lap),
                    "Position": float("ABC".index(driver) + 1),
                    "LapTime_s": lap_time,
                    "Time_s": time,
                    "IsAccurate": not pit,
                    "CleanLap": not pit and lap != 6,
                    "SafetyCar": int(lap == 6),
                    "PitLap": pit,
                    "Degradation_s": 0.1 * lap,
                    "LapsToThreshold": 10 - lap,
                    "Censored": lap == 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def poster_cfg(cfg):
    cfg.processed_dir.mkdir(parents=True)
    tiny_race().to_parquet(cfg.processed_dir / "features_stage1.parquet")
    return cfg


def test_finishing_order_puts_retired_driver_last():
    assert rp.finishing_order(tiny_race()) == ["A", "B", "C"]


def test_stints_and_summary():
    race = tiny_race()
    table = rp.summary_table(race)
    assert table["Driver"].tolist() == ["A", "B", "C"]
    assert table.loc[0, "Stops"] == 1 and table.loc[0, "Tyres"] == "S-H"
    assert table.loc[2, "Laps"] == 5
    assert table.loc[0, "Best lap"] == "1:30.000"  # the 110 s pit lap is not counted
    assert table.loc[0, "Avg clean lap"] == "1:30.000"


def test_gap_to_leader_is_zero_for_the_leader_and_grows():
    gaps = rp.gap_to_leader(tiny_race())
    b = gaps[gaps["Driver"] == "B"].set_index("LapNumber")["Gap_s"]
    assert b[1.0] == pytest.approx(0.5) and b[3.0] == pytest.approx(1.5)
    assert (gaps.groupby("LapNumber")["Gap_s"].min() == 0).all()
    assert (gaps["Gap_s"] >= 0).all()


def test_safety_car_spans_and_facts():
    race = tiny_race()
    assert rp.safety_car_spans(race) == [(6.0, 6.0)]
    facts = rp.race_facts(race)
    assert facts["laps"] == 8 and facts["stops"] == 1 and facts["safety_car_laps"] == 1
    assert facts["fastest_driver"] == "A" and facts["fastest_lap"] == "1:30.000"


def test_lap_text():
    assert rp.lap_text(86.5314) == "1:26.531"
    assert rp.lap_text(59.9996) == "1:00.000"
    assert rp.lap_text(np.nan) == "n/a"


def test_degradation_by_age_uses_clean_laps_only():
    table = rp.degradation_by_age(tiny_race(), min_laps=1)
    soft6 = table[(table["Compound"] == "SOFT") & (table["TyreLife"] == 6.0)]
    assert soft6.empty  # lap 6 was under the safety car
    assert set(table["Compound"]) == {"SOFT", "HARD"}


def test_poster_is_drawn_without_model_scores(poster_cfg):
    path = rp.build_poster(poster_cfg, "2025_01", note="TEST")
    assert path.name == "race_poster_2025_01.png" and path.stat().st_size > 20_000


def test_poster_uses_saved_scores(poster_cfg):
    race = tiny_race()
    pred = race[["RaceID", "Driver", "LapNumber"]].copy()
    pred["lstm_tyre_aware_score"] = 0.2
    pred["lstm_tyre_aware_pred"] = (race["LapNumber"] == 4).astype(int)
    poster_cfg.results_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(poster_cfg.results_dir / "final_predictions.parquet")
    scores = rp.load_scores(poster_cfg, "2025_01", "lstm_tyre_aware")
    assert scores is not None and not scores.attrs["calibrated"] and scores["Call"].sum() == 3
    assert rp.load_scores(poster_cfg, "2025_01", "xgboost") is None
    assert rp.build_poster(poster_cfg, "2025_01").exists()


def test_unknown_race_and_missing_data_give_clear_errors(poster_cfg, tmp_path):
    with pytest.raises(rp.PosterError, match="--list"):
        rp.load_race(poster_cfg, "1999_01")
    (poster_cfg.processed_dir / "features_stage1.parquet").unlink()
    with pytest.raises(rp.PosterError, match="Stage 1"):
        rp.list_races(poster_cfg)
