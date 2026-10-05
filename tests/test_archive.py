import json

import numpy as np
import pandas as pd
import pytest
from synthetic import FakeFastF1

from f1pit import archive
from f1pit.ingest import SourceUnavailableError, ingest, race_sources


def lap_file(driver: str, laps: int = 6, pit_lap: int = 3) -> dict:
    """A driver's file in the archive's own format ('None' for missing values, short column names)."""
    lap = list(range(1, laps + 1))
    return {
        "time": ["None"] + [90.0 + 0.1 * i for i in lap[1:]],
        "lap": lap,
        "compound": ["SOFT" if i <= pit_lap else "HARD" for i in lap],
        "stint": [1 if i <= pit_lap else 2 for i in lap],
        "life": [i if i <= pit_lap else i - pit_lap for i in lap],
        "pos": [1] * laps,
        "status": ["1"] * (laps - 1) + ["4"],
        "sesT": [3600.0 + 90.0 * i for i in lap],
        "pin": [3600.0 + 90.0 * i if i == pit_lap else "None" for i in lap],
        "pout": [3600.0 + 90.0 * i if i == pit_lap + 1 else "None" for i in lap],
        "fresh": [True] * laps,
        "team": ["Team " + driver] * laps,
        "del": [False] * laps,
        "ff1G": [False] * laps,
        "iacc": [False] + [True] * (laps - 1),
        "wAT": [25.0] * laps,
        "wTT": [35.5] * laps,
        "wH": [40.0] * laps,
        "wR": [False] * laps,
    }


def fake_fetch(url: str):
    if "Missing" in url:
        raise archive.ArchiveError(f"not in the archive: {url}")
    if url.endswith("drivers.json"):
        return {"drivers": [{"driver": "AAA"}, {"driver": "BBB"}, {"driver": "DNS"}]}
    if "/DNS/" in url:
        raise archive.ArchiveError(f"not in the archive: {url}")
    return lap_file(url.split("/")[-2])


def test_archive_race_has_the_project_columns_and_types():
    df = archive.load_race_from_archive(2023, 5, "São Paulo Grand Prix", fetch=fake_fetch)
    assert sorted(df["Driver"].unique()) == ["AAA", "BBB"]  # the driver who did not start is skipped
    assert len(df) == 12 and (df["Source"] == "tracinginsights").all()
    assert df["LapTime_s"].isna().sum() == 2 and df["LapTime_s"].dtype == float  # 'None' became NaN
    assert df["PitInTime_s"].notna().sum() == 2 and df["PitOutTime_s"].notna().sum() == 2
    assert df["IsAccurate"].dtype == bool and df["Rainfall"].dtype == bool and df["FreshTyre"].all()
    assert df["TrackStatus"].tolist()[:6] == ["1"] * 5 + ["4"]
    assert (df["Year"] == 2023).all() and (df["Round"] == 5).all() and df["ScheduledLaps"].isna().all()
    assert set(df["Compound"]) == {"SOFT", "HARD"} and df["TrackTemp"].eq(35.5).all()


def test_archive_urls_are_quoted():
    url = archive._url(2023, "São Paulo Grand Prix", "VER/laptimes.json")
    assert " " not in url and "S%C3%A3o%20Paulo%20Grand%20Prix/Race/VER/laptimes.json" in url


def test_archive_file_without_needed_columns_is_refused():
    broken = lap_file("AAA")
    del broken["pin"]
    with pytest.raises(archive.ArchiveError, match="pin"):
        archive.driver_frame("AAA", broken)
    uneven = lap_file("AAA")
    uneven["pos"] = uneven["pos"][:-1]
    with pytest.raises(archive.ArchiveError, match="different lengths"):
        archive.driver_frame("AAA", uneven)


class NamedFastF1(FakeFastF1):
    """The fake FastF1 with event names in the schedule, as the real one has."""

    def get_event_schedule(self, year, include_testing=True):
        schedule = super().get_event_schedule(year, include_testing)
        schedule["EventName"] = [f"Race {r}" for r in schedule["RoundNumber"]]
        return schedule


def test_archive_fills_only_the_races_fastf1_cannot_deliver(cfg):
    calls = []

    def loader(year, rnd, name):
        calls.append((year, rnd, name))
        return archive.load_race_from_archive(year, rnd, name, fetch=fake_fetch)

    f1 = NamedFastF1(rounds=3, broken_races=[(2022, 2)])
    assert ingest(cfg, seasons=[2022], fastf1_module=f1, archive_loader=loader) == []
    assert calls == [(2022, 2, "Race 2")]
    sources = race_sources(cfg).set_index("Source")["Races"].to_dict()
    assert sources == {"fastf1": 2, "tracinginsights": 1}
    saved = pd.read_parquet(cfg.races_dir / "2022_02.parquet")
    assert (saved["Source"] == "tracinginsights").all()
    ingest(cfg, seasons=[2022], fastf1_module=f1, archive_loader=loader)  # nothing is fetched twice
    assert len(calls) == 1


def test_fastf1_is_skipped_after_three_failures_and_the_archive_carries_on(cfg):
    f1 = NamedFastF1(rounds=6, broken_races=[(2022, r) for r in range(1, 7)])

    def loader(year, rnd, name):
        return archive.load_race_from_archive(year, rnd, name, fetch=fake_fetch)

    assert ingest(cfg, seasons=[2022], fastf1_module=f1, archive_loader=loader) == []
    assert f1.download_calls == 3  # rounds 4 to 6 went straight to the archive
    assert len(list(cfg.races_dir.glob("*.parquet"))) == 6


def test_races_missing_from_the_archive_are_listed_and_the_rest_still_load(cfg):
    """The archive answers but lacks rounds 5 and 6 (too recent): record them, keep the other races."""
    f1 = NamedFastF1(rounds=6, broken_races=[(2022, r) for r in range(1, 7)])

    def loader(year, rnd, name):
        name = "Missing " + name if rnd >= 5 else name
        return archive.load_race_from_archive(year, rnd, name, fetch=fake_fetch)

    failures = ingest(cfg, seasons=[2022], fastf1_module=f1, archive_loader=loader)
    assert [(f["year"], f["round"]) for f in failures] == [(2022, 5), (2022, 6)]
    assert len(list(cfg.races_dir.glob("*.parquet"))) == 4
    assert json.loads((cfg.races_dir / "ingest_failures.json").read_text()) == failures


def test_download_stops_when_no_source_can_be_reached(cfg):
    f1 = NamedFastF1(rounds=6, broken_races=[(2022, r) for r in range(1, 7)])

    def loader(year, rnd, name):
        raise archive.ArchiveError("could not read https://example.invalid: timed out")

    with pytest.raises(SourceUnavailableError, match="any source"):
        ingest(cfg, seasons=[2022], fastf1_module=f1, archive_loader=loader)
    failures = json.loads((cfg.races_dir / "ingest_failures.json").read_text())
    assert len(failures) == 3 and "archive" in failures[0]["error"]


def test_archive_races_pass_through_cleaning(cfg):
    from f1pit.clean import clean

    df = archive.load_race_from_archive(2022, 1, "Race 1", fetch=fake_fetch)
    out, report = clean(df, cfg)
    assert report["pit_laps"] == 2 and out["TotalLaps"].eq(6).all()
    assert np.isclose(out["SafetyCar"].mean(), 1 / 6)
