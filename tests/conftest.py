import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import FakeFastF1  # noqa: E402

from f1pit.clean import clean  # noqa: E402
from f1pit.config import load_config  # noqa: E402
from f1pit.ingest import ingest, read_all_races  # noqa: E402

CONFIG_TEXT = """
seasons: {train: [2022, 2023], valid: [2024], test: [2025], extra_test: []}
quick: false
quick_races: 4
random_seed: 42
paths: {cache: data/cache, races: data/races, processed: data/processed, results: results}
rule_forced_races: [[2023, Qatar]]
clean: {min_laps_for_fuel_fit: 100, stint_baseline_laps: 3}
"""


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "test.yaml"
    path.write_text(CONFIG_TEXT)
    return load_config(path, root=tmp_path)


@pytest.fixture
def fake_f1():
    # 2023 round 2 is wet; 2024 round 3 fails to download
    return FakeFastF1(rounds=6, wet_races=[(2023, 2)], broken_races=[(2024, 3)])


@pytest.fixture
def raw(cfg, fake_f1):
    ingest(cfg, seasons=[2022, 2023, 2024], fastf1_module=fake_f1)
    return read_all_races(cfg)


@pytest.fixture
def laps(raw, cfg):
    df, _ = clean(raw, cfg)
    return df
