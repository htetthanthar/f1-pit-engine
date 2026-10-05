import json

import pandas as pd
from synthetic import FakeFastF1, FakeSession

from f1pit.ingest import completed_rounds, ingest, load_race, session_to_frame


def test_session_to_frame_has_plain_columns():
    s = FakeSession(2022, 1)
    s.load()
    df = session_to_frame(s, 2022, 1)
    for col in [
        "Driver",
        "LapNumber",
        "Compound",
        "TyreLife",
        "LapTime_s",
        "Time_s",
        "PitInTime_s",
        "PitOutTime_s",
        "TrackTemp",
        "Year",
        "Round",
        "EventName",
    ]:
        assert col in df.columns
    assert df["LapTime_s"].between(60, 200).all()
    assert df["FreshTyre"].dtype == bool and df["Deleted"].dtype == bool  # None/NaN became False
    assert (df["EventName"] == "Bahrain Grand Prix").all()


def test_failed_download_is_reported_not_crashed(cfg, fake_f1):
    failures = ingest(cfg, seasons=[2024], fastf1_module=fake_f1)
    assert [(f["year"], f["round"]) for f in failures] == [(2024, 3)]
    saved = json.loads((cfg.races_dir / "ingest_failures.json").read_text())
    assert saved[0]["round"] == 3


def test_saved_races_are_not_downloaded_again(cfg):
    f1 = FakeFastF1(rounds=2)
    ingest(cfg, seasons=[2022], fastf1_module=f1)
    first = f1.download_calls
    ingest(cfg, seasons=[2022], fastf1_module=f1)
    assert first == 2 and f1.download_calls == 2
    assert isinstance(load_race(cfg, 2022, 1, f1), pd.DataFrame)


def test_future_races_are_skipped():
    schedule = FakeFastF1(rounds=3, future_rounds=2).get_event_schedule(2026)
    assert completed_rounds(schedule) == [1, 2, 3]


def test_quick_mode_limits_races(cfg):
    from dataclasses import replace

    quick_cfg = replace(cfg, quick=True, quick_races=2)
    f1 = FakeFastF1(rounds=6)
    ingest(quick_cfg, seasons=[2022], fastf1_module=f1)
    assert f1.download_calls == 2


def test_scheduled_race_distance_is_saved():
    s = FakeSession(2022, 1)
    s.load()
    assert (session_to_frame(s, 2022, 1)["ScheduledLaps"] == 55).all()
    s.total_laps = None  # FastF1 has no lap-count data for this race
    assert session_to_frame(s, 2022, 1)["ScheduledLaps"].isna().all()


def test_download_stops_when_the_source_gives_no_data(cfg):
    """Three races in a row with no data means the service is refusing us: stop and explain."""
    import pytest

    from f1pit.ingest import SourceUnavailableError

    working = FakeFastF1(rounds=2)
    ingest(cfg, seasons=[2022], fastf1_module=working)  # 2022 was downloaded earlier and is saved

    blocked = FakeFastF1(rounds=6, broken_races=[(y, r) for y in (2022, 2023) for r in range(1, 7)])
    with pytest.raises(SourceUnavailableError, match="your own computer"):
        ingest(cfg, seasons=[2022, 2023], fastf1_module=blocked)
    assert blocked.download_calls == 3  # rounds 1 and 2 came from disk; it gave up after rounds 3, 4, 5
    saved = json.loads((cfg.races_dir / "ingest_failures.json").read_text())
    assert [(f["year"], f["round"]) for f in saved] == [(2022, 3), (2022, 4), (2022, 5)]
    assert len(list(cfg.races_dir.glob("*.parquet"))) == 2  # saved races are untouched
