"""Load and check the project settings file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    train_seasons: list[int]
    valid_seasons: list[int]
    test_seasons: list[int]
    extra_test_seasons: list[int]
    quick: bool
    quick_races: int
    random_seed: int
    cache_dir: Path
    races_dir: Path
    processed_dir: Path
    results_dir: Path
    rule_forced_races: list[tuple[int, str]]
    min_laps_for_fuel_fit: int
    stint_baseline_laps: int
    # Stage 1 (optional section in the YAML; these are the defaults)
    thresholds_s: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    primary_threshold_s: float = 1.0
    horizon_laps: int = 40
    forecast_horizons: list[int] = field(default_factory=lambda: [1, 5, 10])
    oof_folds: int = 5
    event_min_laps: int = 200
    min_relative_improvement: float = 0.01
    # Stage 2 settings are kept as a plain dict and checked in f1pit.stage2
    stage2: dict = field(default_factory=dict)
    # Final-evaluation settings, checked in f1pit.evaluate
    evaluation: dict = field(default_factory=dict)

    @property
    def all_seasons(self) -> list[int]:
        return self.train_seasons + self.valid_seasons + self.test_seasons + self.extra_test_seasons

    def split_of(self, year: int) -> str:
        """Name of the data split a season belongs to."""
        for name, seasons in (
            ("train", self.train_seasons),
            ("valid", self.valid_seasons),
            ("test", self.test_seasons),
            ("extra_test", self.extra_test_seasons),
        ):
            if year in seasons:
                return name
        return "unused"


def load_config(path: str | Path = "configs/default.yaml", root: str | Path | None = None) -> Config:
    """Read the YAML settings. Relative data paths are resolved against `root` (default: the file's repo)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = Path(root) if root is not None else path.resolve().parent.parent

    seasons = raw["seasons"]
    groups = [seasons["train"], seasons["valid"], seasons["test"], seasons.get("extra_test", [])]
    flat = [y for g in groups for y in g]
    if len(flat) != len(set(flat)):
        raise ValueError("A season appears in more than one split; splits must not overlap.")

    paths = {k: (root / v) for k, v in raw["paths"].items()}
    return Config(
        train_seasons=list(seasons["train"]),
        valid_seasons=list(seasons["valid"]),
        test_seasons=list(seasons["test"]),
        extra_test_seasons=list(seasons.get("extra_test", [])),
        quick=bool(raw.get("quick", False)),
        quick_races=int(raw.get("quick_races", 4)),
        random_seed=int(raw.get("random_seed", 42)),
        cache_dir=paths["cache"],
        races_dir=paths["races"],
        processed_dir=paths["processed"],
        results_dir=paths["results"],
        rule_forced_races=[(int(y), str(k)) for y, k in raw.get("rule_forced_races", [])],
        min_laps_for_fuel_fit=int(raw["clean"]["min_laps_for_fuel_fit"]),
        stint_baseline_laps=int(raw["clean"]["stint_baseline_laps"]),
        **_stage1_settings(raw.get("stage1", {})),
        stage2=dict(raw.get("stage2") or {}),
        evaluation=dict(raw.get("evaluation") or {}),
    )


def _stage1_settings(s1: dict) -> dict:
    out = {}
    if "thresholds_s" in s1:
        out["thresholds_s"] = [float(t) for t in s1["thresholds_s"]]
    for key, cast in (
        ("primary_threshold_s", float),
        ("horizon_laps", int),
        ("oof_folds", int),
        ("event_min_laps", int),
        ("min_relative_improvement", float),
    ):
        if key in s1:
            out[key] = cast(s1[key])
    if "forecast_horizons" in s1:
        out["forecast_horizons"] = [int(h) for h in s1["forecast_horizons"]]
    primary = out.get("primary_threshold_s", 1.0)
    if primary not in out.get("thresholds_s", [0.5, 1.0, 1.5]):
        raise ValueError("stage1.primary_threshold_s must be one of stage1.thresholds_s")
    return out
