import pandas as pd
from synthetic import RETIRE_LAP, RETIRING_DRIVER

from f1pit.clean import clean


def test_wet_race_is_removed(raw, cfg):
    df, report = clean(raw, cfg)
    assert report["wet_races_removed"] == ["2023_02"]
    assert "2023_02" not in set(df["RaceID"])
    assert set(df["Compound"]) <= {"SOFT", "MEDIUM", "HARD"}


def test_retirement_into_pits_is_not_a_pit_stop(laps):
    last = laps[(laps["Driver"] == RETIRING_DRIVER) & (laps["LapNumber"] == RETIRE_LAP)]
    assert len(last) > 0
    assert last["PitInTime_s"].notna().all()  # the car did enter the pits...
    assert (last["PitLap"] == 0).all()  # ...but it is not labelled as a tyre stop


def test_every_tyre_stop_is_labelled_exactly_once(laps):
    per_driver = laps.groupby(["RaceID", "Driver"]).agg(stops=("PitLap", "sum"), stints=("Stint", "max"))
    assert (per_driver["stops"] == per_driver["stints"] - 1).all()


def test_rule_forced_race_is_flagged(laps):
    flagged = set(laps.loc[laps["RuleForced"], "RaceID"])
    assert flagged == {"2023_04"}  # round 4 is the Qatar Grand Prix in the fake calendar
    assert not laps.loc[laps["Year"] == 2022, "RuleForced"].any()  # Qatar 2022 is not in the list


def test_safety_car_flag_and_splits(laps):
    sc = laps[laps["TrackStatus"] == "4"]
    assert (sc["SafetyCar"] == 1).all()
    assert set(laps["Split"]) == {"train", "valid"}
    assert laps["RaceProgress"].between(0, 1).all()


def test_race_length_is_the_scheduled_distance(raw, cfg):
    """A race stopped early must not reveal its real length: progress uses the scheduled distance."""
    race = (raw["Year"] == 2022) & (raw["Round"] == 1)
    stopped = raw[~(race & (raw["LapNumber"] > 40))]  # 2022 round 1 red-flagged after 40 of 55 laps
    df, _ = clean(stopped, cfg)
    short = df[df["RaceID"] == "2022_01"]
    assert short["LapNumber"].max() == 40 and (short["TotalLaps"] == 55).all()
    assert short["LapsRemaining"].min() == 15 and short["RaceProgress"].max() < 0.75

    old_files = stopped.drop(columns="ScheduledLaps")  # races saved before the column existed
    assert (clean(old_files, cfg)[0].query("RaceID == '2022_01'")["TotalLaps"] == 40).all()
    wrong = stopped.assign(ScheduledLaps=10.0)  # a scheduled distance below the laps run is ignored
    assert (clean(wrong, cfg)[0].query("RaceID == '2022_01'")["TotalLaps"] == 40).all()


def test_a_race_copied_into_another_season_is_refused(raw, cfg):
    """Copying 2022 round 1 to 2024 round 1 and adding noise to the lap times must stop the pipeline."""
    import numpy as np
    import pandas as pd
    import pytest

    from f1pit.clean import CopiedRaceError, copied_races

    assert copied_races(clean(raw, cfg)[0]) == []
    fake = raw[(raw["Year"] == 2022) & (raw["Round"] == 1)].copy()
    fake["Year"] = 2024
    fake["LapTime_s"] += np.random.default_rng(0).normal(0, 0.15, len(fake))
    fake["Time_s"] += np.random.default_rng(1).normal(0, 0.15, len(fake))
    mixed = pd.concat([raw[~((raw["Year"] == 2024) & (raw["Round"] == 1))], fake], ignore_index=True)
    with pytest.raises(CopiedRaceError, match="2024_01 .same laps as 2022_01"):
        clean(mixed, cfg)


def _car(pit_in=None, pit_out_next=None, life_after=1.0, compound_after="HARD"):
    """One car, six laps, entering the pits on lap 3."""
    rows = []
    for lap in range(1, 7):
        after = lap > 3
        rows.append(
            {
                "RaceID": "2024_01",
                "Driver": "AAA",
                "LapNumber": float(lap),
                "Compound": compound_after if after else "SOFT",
                "TyreLife": life_after + (lap - 4) if after else float(lap),
                "PitInTime_s": pit_in if lap == 3 else float("nan"),
                "PitOutTime_s": pit_out_next if lap == 4 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_only_tyre_stops_made_as_a_race_decision_are_pit_laps():
    from f1pit.clean import label_pit_laps

    stop = label_pit_laps(_car(pit_in=1000.0, pit_out_next=1023.0))
    assert stop["PitLap"].tolist() == [0, 0, 1, 0, 0, 0]

    used_set = label_pit_laps(_car(pit_in=1000.0, pit_out_next=1023.0, life_after=9.0, compound_after="SOFT"))
    assert used_set["PitLap"].sum() == 1  # same compound but a different (used) set: still a tyre stop

    through = label_pit_laps(_car(pit_in=1000.0, pit_out_next=1016.0, life_after=4.0, compound_after="SOFT"))
    assert through["PitLap"].sum() == 0 and through["PitLaneNoChange"].tolist() == [0, 0, 1, 0, 0, 0]

    red_flag = label_pit_laps(_car(pit_in=1000.0, pit_out_next=2600.0))
    assert red_flag["PitLap"].sum() == 0 and red_flag["PitSuspended"].sum() == 1

    no_exit_time = label_pit_laps(_car(pit_in=1000.0, pit_out_next=None))
    assert no_exit_time["PitLap"].sum() == 1  # unknown pit time is not treated as a suspension
