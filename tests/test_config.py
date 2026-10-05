import pytest

from f1pit.config import load_config


def test_splits_are_named(cfg):
    assert cfg.split_of(2022) == "train"
    assert cfg.split_of(2024) == "valid"
    assert cfg.split_of(2025) == "test"
    assert cfg.split_of(2019) == "unused"


def test_overlapping_seasons_are_rejected(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "seasons: {train: [2022, 2024], valid: [2024], test: [2025]}\n"
        "paths: {cache: a, races: b, processed: c, results: d}\n"
        "clean: {min_laps_for_fuel_fit: 100, stint_baseline_laps: 3}\n"
    )
    with pytest.raises(ValueError, match="more than one split"):
        load_config(tmp_path / "c.yaml", root=tmp_path)


def test_default_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.train_seasons == [2022, 2023]
    assert set(cfg.train_seasons).isdisjoint(cfg.test_seasons + cfg.extra_test_seasons)
