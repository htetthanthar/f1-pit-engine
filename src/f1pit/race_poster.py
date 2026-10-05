"""One-page race analysis poster for a single race.

    python -m f1pit.race_poster --list
    python -m f1pit.race_poster --race 2025_05

The poster is drawn only from this project's own files, so every panel can be traced to a column:

  * data/processed/features_stage1.parquet   lap times, positions, tyres, degradation, Stage 1 forecast
  * results/final_predictions.parquet         Stage 2 scores for the held-out test races
  * results/stage2_validation_predictions.parquet   Stage 2 scores for the validation races

It deliberately has no tyre-temperature, top-speed or driver-rating panel: the public timing data used
here does not contain tyre temperatures, speed-trap values are not downloaded by this project, and a
rating would have to be invented. Races used for training have no model-score panel, because a model's
score on a race it learned from says nothing about how good it is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from f1pit.config import Config, load_config  # noqa: E402
from f1pit.plots import LABELS  # noqa: E402

TOP_N = 5
TABLE_ROWS = 10

SURFACE = "#1a1a19"
PANEL = "#222221"
INK = "#ffffff"
INK_SOFT = "#c3c2b7"
INK_MUTED = "#8f8e85"
GRID = "#3a3a37"
SAFETY_CAR = "#fab219"
# Fixed order: the colour belongs to the finishing place within this poster (1st to 5th).
DRIVER_COLOURS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
# The sport's own tyre colours (red soft, yellow medium, white hard); every bar also carries a letter.
COMPOUND_COLOURS = {
    "SOFT": "#e66767",
    "MEDIUM": "#e0b030",
    "HARD": "#e8e7df",
    "INTERMEDIATE": "#0ca30c",
    "WET": "#3987e5",
}
UNKNOWN_COMPOUND = "#6b6a63"
SCORE_MAP = LinearSegmentedColormap.from_list("score", ["#1d2734", "#2a5d9e", "#3987e5", "#d6e6fb"])


class PosterError(RuntimeError):
    """The poster cannot be drawn; the message says what is missing."""


# ----------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------
def _features(cfg: Config) -> pd.DataFrame:
    path = cfg.processed_dir / "features_stage1.parquet"
    if not path.exists():
        raise PosterError(f"{path} not found. Run the pipeline up to Stage 1 first (python -m f1pit.stage1).")
    return pd.read_parquet(path)


def list_races(cfg: Config) -> pd.DataFrame:
    """Every race that can be drawn: id, name, split."""
    df = _features(cfg)
    out = df.groupby("RaceID", sort=True).agg(EventName=("EventName", "first"), Split=("Split", "first"))
    return out.reset_index()


def load_race(cfg: Config, race_id: str) -> pd.DataFrame:
    df = _features(cfg)
    race = df[df["RaceID"] == race_id].copy()
    if race.empty:
        raise PosterError(
            f"Race {race_id!r} is not in the processed data. Use --list to see the races available "
            "(wet races are removed when the data is cleaned)."
        )
    return race.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def load_scores(cfg: Config, race_id: str, model: str) -> pd.DataFrame | None:
    """Stage 2 output for this race, or None when the race was used for training or nothing is saved.

    Returns Driver, LapNumber, Score (0-1 when `IsProbability`), Call (the model's pit decision).
    """
    for name in ("final_predictions.parquet", "stage2_validation_predictions.parquet"):
        path = cfg.results_dir / name
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        pred = pred[pred["RaceID"] == race_id]
        if pred.empty or f"{model}_score" not in pred.columns:
            continue
        calibrated = f"{model}_calibrated"
        use = (
            calibrated if calibrated in pred.columns and pred[calibrated].notna().any() else f"{model}_score"
        )
        out = pred[["Driver", "LapNumber"]].copy()
        out["Score"] = pred[use].astype(float)
        out["Call"] = pred[f"{model}_pred"].fillna(0).astype(int)
        out.attrs["calibrated"] = use == calibrated
        out.attrs["source"] = name
        return out.reset_index(drop=True)
    return None


def finishing_order(race: pd.DataFrame) -> list[str]:
    """Drivers from first to last: most laps completed first, then position on their final lap."""
    last = race.sort_values("LapNumber").groupby("Driver").tail(1)
    last = last.assign(Position=last["Position"].fillna(np.inf))
    last = last.sort_values(["LapNumber", "Position", "Driver"], ascending=[False, True, True])
    return last["Driver"].tolist()


def stints(race: pd.DataFrame) -> pd.DataFrame:
    """One row per driver stint: first lap, last lap, compound."""
    out = race.groupby(["Driver", "Stint"], sort=True).agg(
        First=("LapNumber", "min"), Last=("LapNumber", "max"), Compound=("Compound", "first")
    )
    return out.reset_index()


def gap_to_leader(race: pd.DataFrame) -> pd.DataFrame:
    """Seconds behind whichever car had completed that lap first."""
    out = race[["Driver", "LapNumber", "Time_s"]].dropna().copy()
    out["Gap_s"] = out["Time_s"] - out.groupby("LapNumber")["Time_s"].transform("min")
    return out


def degradation_by_age(race: pd.DataFrame, min_laps: int = 3) -> pd.DataFrame:
    """Median and quartiles of fuel-corrected pace loss at each tyre age, per compound (clean laps only)."""
    ok = race[race["CleanLap"].astype(bool) & race["Degradation_s"].notna()]
    grouped = ok.groupby(["Compound", "TyreLife"])["Degradation_s"]
    out = grouped.agg(
        Median="median", Low=lambda s: s.quantile(0.25), High=lambda s: s.quantile(0.75), N="size"
    )
    return out[out["N"] >= min_laps].reset_index()


def safety_car_spans(race: pd.DataFrame) -> list[tuple[float, float]]:
    """Runs of consecutive laps on which any car ran under a safety car or virtual safety car."""
    laps = sorted(race.loc[race["SafetyCar"] == 1, "LapNumber"].unique())
    spans: list[tuple[float, float]] = []
    for lap in laps:
        if spans and lap == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], lap)
        else:
            spans.append((lap, lap))
    return spans


def lap_text(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    total = round(float(seconds), 3)
    minutes = int(total // 60)
    return f"{minutes}:{total - 60 * minutes:06.3f}"


def summary_table(race: pd.DataFrame, rows: int = TABLE_ROWS) -> pd.DataFrame:
    order = finishing_order(race)[:rows]
    stint_table = stints(race)
    out = []
    for place, driver in enumerate(order, start=1):
        laps = race[race["Driver"] == driver]
        clean = laps.loc[laps["CleanLap"].astype(bool), "LapTime_s"]
        strategy = stint_table.loc[stint_table["Driver"] == driver, "Compound"]
        out.append(
            {
                "Pos": place,
                "Driver": driver,
                "Team": str(laps["Team"].iloc[0]),
                "Laps": int(laps["LapNumber"].max()),
                "Best lap": lap_text(laps.loc[laps["IsAccurate"].astype(bool), "LapTime_s"].min()),
                "Avg clean lap": lap_text(clean.mean() if len(clean) else np.nan),
                "Stops": int(laps["PitLap"].sum()),
                "Tyres": "-".join(str(c)[:1] if isinstance(c, str) and c else "?" for c in strategy),
            }
        )
    return pd.DataFrame(out)


def race_facts(race: pd.DataFrame) -> dict:
    accurate = race[race["IsAccurate"].astype(bool) & race["LapTime_s"].notna()]
    fastest = accurate.loc[accurate["LapTime_s"].idxmin()] if len(accurate) else None
    return {
        "laps": int(race["LapNumber"].max()),
        "drivers": int(race["Driver"].nunique()),
        "stops": int(race["PitLap"].sum()),
        "safety_car_laps": int(race.loc[race["SafetyCar"] == 1, "LapNumber"].nunique()),
        "fastest_driver": None if fastest is None else str(fastest["Driver"]),
        "fastest_lap": None if fastest is None else lap_text(fastest["LapTime_s"]),
        "fastest_lap_number": None if fastest is None else int(fastest["LapNumber"]),
        "rule_forced": bool(race["RuleForced"].max()),
        "split": str(race["Split"].iloc[0]),
    }


# ----------------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------------
def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=11)
    ax.tick_params(colors=INK_SOFT, labelsize=10, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _shade_safety_car(ax, spans) -> None:
    for first, last in spans:
        ax.axvspan(first - 0.5, last + 0.5, color=SAFETY_CAR, alpha=0.13, linewidth=0, zorder=0)


def _legend(ax, handles, **kwargs) -> None:
    if not handles:
        return
    legend = ax.legend(handles=handles, frameon=False, fontsize=10, labelcolor=INK_SOFT, **kwargs)
    legend.set_zorder(5)


def _driver_handles(top: list[str], spans) -> list:
    handles = [
        Line2D([], [], color=DRIVER_COLOURS[i], linewidth=2.5, label=f"P{i + 1}  {d}")
        for i, d in enumerate(top)
    ]
    if spans:
        handles.append(Patch(facecolor=SAFETY_CAR, alpha=0.3, label="Safety car / VSC"))
    return handles


def _below(ax, handles, ncol: int | None = None) -> None:
    _legend(ax, handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol or len(handles))


def _lap_times(ax, race, top, spans) -> None:
    _style(ax, f"Lap time, top {len(top)} finishers", "Lap", "Lap time (s)")
    _shade_safety_car(ax, spans)
    values = []
    for i, driver in enumerate(top):
        laps = race[race["Driver"] == driver]
        ax.plot(laps["LapNumber"], laps["LapTime_s"], color=DRIVER_COLOURS[i], linewidth=2, zorder=3)
        values.append(laps.loc[laps["CleanLap"].astype(bool), "LapTime_s"])
    clean = pd.concat(values).dropna() if values else pd.Series(dtype=float)
    if len(clean):
        # Zoom on racing laps; pit and safety-car laps run off the top and are marked by the legend note.
        low, high = clean.quantile(0.01), clean.quantile(0.99)
        pad = max(0.3, 0.15 * (high - low))
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(0.5, race["LapNumber"].max() + 0.5)
    _below(ax, _driver_handles(top, spans))
    ax.text(
        1.0, 1.02, "Pit-stop and safety-car laps run off the top of the scale", transform=ax.transAxes,
        ha="right", va="bottom", color=INK_MUTED, fontsize=9,
    )  # fmt: skip


def _strategy(ax, race, order) -> None:
    _style(ax, "Tyre strategy, in finishing order", "Lap")
    table = stints(race)
    used = []
    for row, driver in enumerate(order):
        for stint in table[table["Driver"] == driver].itertuples():
            compound = stint.Compound if isinstance(stint.Compound, str) else "UNKNOWN"
            colour = COMPOUND_COLOURS.get(compound, UNKNOWN_COMPOUND)
            if compound not in used:
                used.append(compound)
            width = stint.Last - stint.First + 1
            ax.barh(row, width - 0.25, left=stint.First - 0.5 + 0.125, height=0.68, color=colour, zorder=3)
            if width >= 4:
                ax.text(
                    stint.First - 0.5 + width / 2, row, compound[:1], ha="center", va="center",
                    color=SURFACE, fontsize=10, fontweight="bold", zorder=4,
                )  # fmt: skip
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{i + 1:>2}  {d}" for i, d in enumerate(order)], fontsize=9)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0.5, race["LapNumber"].max() + 0.5)
    ax.grid(False, axis="y")
    names = {
        "SOFT": "Soft (S)",
        "MEDIUM": "Medium (M)",
        "HARD": "Hard (H)",
        "INTERMEDIATE": "Intermediate (I)",
    }
    handles = [
        Patch(facecolor=COMPOUND_COLOURS.get(c, UNKNOWN_COMPOUND), label=names.get(c, c.title()))
        for c in COMPOUND_COLOURS
        if c in used
    ]
    handles += [
        Patch(facecolor=UNKNOWN_COMPOUND, label="Unknown") for c in used if c not in COMPOUND_COLOURS
    ][:1]
    _below(ax, handles)


def _positions(ax, race, order, top, spans) -> None:
    _style(ax, "Position by lap", "Lap", "Position")
    _shade_safety_car(ax, spans)
    for driver in order:
        if driver in top:
            continue
        laps = race[race["Driver"] == driver]
        ax.plot(laps["LapNumber"], laps["Position"], color=INK_MUTED, linewidth=1, alpha=0.45, zorder=2)
    for i, driver in enumerate(top):
        laps = race[race["Driver"] == driver]
        ax.plot(laps["LapNumber"], laps["Position"], color=DRIVER_COLOURS[i], linewidth=2.2, zorder=3)
    worst = int(np.nanmax(race["Position"])) if race["Position"].notna().any() else len(order)
    ax.set_ylim(worst + 0.5, 0.5)
    ax.set_yticks([1] + list(range(5, worst + 1, 5)))
    ax.set_xlim(0.5, race["LapNumber"].max() + 0.5)
    handles = _driver_handles(top, spans)
    handles.insert(len(top), Line2D([], [], color=INK_MUTED, linewidth=1, alpha=0.7, label="Other drivers"))
    _below(ax, handles, ncol=4)


def _gaps(ax, race, top, spans) -> None:
    _style(ax, "Gap to the race leader", "Lap", "Seconds behind the leader")
    _shade_safety_car(ax, spans)
    gaps = gap_to_leader(race)
    for i, driver in enumerate(top):
        g = gaps[gaps["Driver"] == driver]
        ax.plot(g["LapNumber"], g["Gap_s"], color=DRIVER_COLOURS[i], linewidth=2, zorder=3)
    shown = gaps[gaps["Driver"].isin(top)]["Gap_s"]
    ax.set_ylim((shown.max() if len(shown) else 1) * 1.05 + 0.5, -0.5)
    ax.set_xlim(0.5, race["LapNumber"].max() + 0.5)
    _below(ax, _driver_handles(top, spans))


def _degradation(ax, race, threshold: float) -> None:
    _style(ax, "Tyre degradation by tyre age", "Tyre age (laps)", "Pace loss vs start of stint (s)")
    table = degradation_by_age(race)
    handles = []
    for compound, colour in COMPOUND_COLOURS.items():
        part = table[table["Compound"] == compound].sort_values("TyreLife")
        if part.empty:
            continue
        ax.fill_between(
            part["TyreLife"], part["Low"], part["High"], color=colour, alpha=0.16, linewidth=0, zorder=2
        )
        ax.plot(part["TyreLife"], part["Median"], color=colour, linewidth=2.2, zorder=3)
        handles.append(Line2D([], [], color=colour, linewidth=2.5, label=f"{compound.title()} (median)"))
    ax.axhline(threshold, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    handles.append(
        Line2D(
            [], [], color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)), label=f"{threshold:g} s threshold"
        )
    )
    if not table.empty:
        ax.set_xlim(table["TyreLife"].min() - 0.5, table["TyreLife"].max() + 0.5)
    _below(ax, handles)
    ax.text(
        1.0, 1.02, "Fuel-corrected, clean laps only. Band = middle half of the cars", transform=ax.transAxes,
        ha="right", va="bottom", color=INK_MUTED, fontsize=9,
    )  # fmt: skip


def _forecast(ax, race, winner: str, threshold: float, horizon: int, spans) -> None:
    _style(
        ax, f"Stage 1 forecast for the winner ({winner})", "Lap",
        f"Forecast laps until {threshold:g} s pace loss",
    )  # fmt: skip
    _shade_safety_car(ax, spans)
    laps = race[race["Driver"] == winner]
    for _, stint in laps.groupby("Stint"):
        colour = COMPOUND_COLOURS.get(stint["Compound"].iloc[0], UNKNOWN_COMPOUND)
        ax.plot(stint["LapNumber"], stint["LapsToThreshold"], color=colour, linewidth=2.2, zorder=3)
        capped = stint[stint["Censored"].astype(bool)]
        ax.scatter(
            capped["LapNumber"], capped["LapsToThreshold"], s=26, facecolor=PANEL, edgecolor=colour, zorder=4
        )
    stops = laps.loc[laps["PitLap"] == 1, "LapNumber"]
    for lap in stops:
        ax.axvline(lap, color=INK, linewidth=1.2, linestyle=(0, (2, 3)), zorder=2)
    ax.set_ylim(-1, horizon + 2)
    ax.set_xlim(0.5, race["LapNumber"].max() + 0.5)
    names = [c for c in COMPOUND_COLOURS if c in set(laps["Compound"])]
    handles = [
        Line2D([], [], color=COMPOUND_COLOURS[c], linewidth=2.5, label=f"On {c.title()}") for c in names
    ]
    handles.append(Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (2, 3)), label="Real pit stop"))
    if laps["Censored"].astype(bool).any():
        handles.append(
            Line2D([], [], marker="o", linestyle="", markerfacecolor=PANEL, markeredgecolor=INK_SOFT,
                   label=f"Not reached within {horizon} laps")
        )  # fmt: skip
    _below(ax, handles)


def _scores(fig, ax, race, top, scores: pd.DataFrame | None, model: str) -> None:
    label = LABELS.get(model, model)
    _style(ax, f"Stage 2 pit decision: {label}", "Lap")
    ax.grid(False)
    if scores is None:
        split = str(race["Split"].iloc[0])
        reason = (
            "This race was used to train the models,\nso its scores are not shown."
            if split == "train"
            else "No saved model scores for this race yet.\nRun Stage 2 (and the final test for test races)."
        )
        ax.text(
            0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", color=INK_SOFT, fontsize=12
        )
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    total = int(race["LapNumber"].max())
    grid = np.full((len(top), total), np.nan)
    for row, driver in enumerate(top):
        part = scores[scores["Driver"] == driver]
        laps = part["LapNumber"].to_numpy(int)
        keep = (laps >= 1) & (laps <= total)
        grid[row, laps[keep] - 1] = part["Score"].to_numpy(float)[keep]
    is_probability = bool(scores.attrs.get("calibrated")) or (np.nanmin(grid) >= 0 and np.nanmax(grid) <= 1)
    cmap = SCORE_MAP.with_extremes(bad=PANEL)
    image = ax.imshow(
        np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, extent=(0.5, total + 0.5, len(top) - 0.5, -0.5),
        vmin=0 if is_probability else None, vmax=1 if is_probability else None, interpolation="nearest",
    )  # fmt: skip
    for row, driver in enumerate(top):
        real = race[(race["Driver"] == driver) & (race["PitLap"] == 1)]["LapNumber"]
        ax.scatter(
            real,
            [row - 0.27] * len(real),
            marker="v",
            s=90,
            color=INK,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=4,
        )
        calls = scores[(scores["Driver"] == driver) & (scores["Call"] == 1)]["LapNumber"]
        ax.scatter(
            calls,
            [row + 0.27] * len(calls),
            marker="^",
            s=60,
            color="#d95926",
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=4,
        )
    for row in range(1, len(top)):
        ax.axhline(row - 0.5, color=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"P{i + 1}  {d}" for i, d in enumerate(top)], fontsize=10)
    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.035)
    kind = "Calibrated pit probability" if scores.attrs.get("calibrated") else "Pit score"
    bar.set_label(kind, color=INK_SOFT, fontsize=10)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    bar.outline.set_visible(False)
    handles = [
        Line2D([], [], marker="v", linestyle="", color=INK, markersize=9, label="Real pit stop"),
        Line2D([], [], marker="^", linestyle="", color="#d95926", markersize=8, label="Model says pit"),
    ]
    _below(ax, handles)
    if np.isnan(grid).any():
        ax.text(
            1.0, 1.02, "Blank = no prediction saved for that lap", transform=ax.transAxes, ha="right",
            va="bottom", color=INK_MUTED, fontsize=9,
        )  # fmt: skip


def _table(ax, race) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title("Race summary", color=INK, fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    data = summary_table(race)
    widths = [0.06, 0.10, 0.22, 0.08, 0.14, 0.16, 0.09, 0.15]
    table = ax.table(
        cellText=data.astype(str).values.tolist(), colLabels=list(data.columns), colWidths=widths,
        cellLoc="center", bbox=[0.02, 0.03, 0.96, 0.94],
    )  # fmt: skip
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#33332f")
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_facecolor(PANEL if row % 2 else "#272725")
            cell.set_text_props(color=INK if col <= 1 else INK_SOFT)


def build_poster(
    cfg: Config,
    race_id: str,
    out: Path | None = None,
    model: str = "lstm_tyre_aware",
    note: str | None = None,
) -> Path:
    """Draw the poster for one race and return the PNG path."""
    race = load_race(cfg, race_id)
    scores = load_scores(cfg, race_id, model)
    order = finishing_order(race)
    top = order[:TOP_N]
    spans = safety_car_spans(race)
    facts = race_facts(race)
    threshold = cfg.primary_threshold_s

    fig = plt.figure(figsize=(21, 27), facecolor=SURFACE)
    grid = fig.add_gridspec(4, 2, left=0.055, right=0.975, top=0.895, bottom=0.075, hspace=0.42, wspace=0.16)
    _lap_times(fig.add_subplot(grid[0, 0]), race, top, spans)
    _strategy(fig.add_subplot(grid[0, 1]), race, order)
    _positions(fig.add_subplot(grid[1, 0]), race, order, top, spans)
    _gaps(fig.add_subplot(grid[1, 1]), race, top, spans)
    _degradation(fig.add_subplot(grid[2, 0]), race, threshold)
    _forecast(fig.add_subplot(grid[2, 1]), race, top[0], threshold, cfg.horizon_laps, spans)
    _scores(fig, fig.add_subplot(grid[3, 0]), race, top, scores, model)
    _table(fig.add_subplot(grid[3, 1]), race)

    year = int(race["Year"].iloc[0])
    name = str(race["EventName"].iloc[0])
    fig.text(0.055, 0.976, f"{name} {year}", color=INK, fontsize=30, fontweight="bold", va="center")
    fig.text(
        0.055,
        0.956,
        "Race analysis: pace, tyres and the pit-stop models",
        color=INK_SOFT,
        fontsize=15,
        va="center",
    )
    parts = [
        f"{facts['laps']} laps",
        f"{facts['drivers']} drivers",
        f"{facts['stops']} pit stops",
        f"{facts['safety_car_laps']} laps under safety car or VSC",
    ]
    if facts["fastest_driver"]:
        parts.append(
            f"fastest lap {facts['fastest_lap']} "
            f"({facts['fastest_driver']}, lap {facts['fastest_lap_number']})"
        )
    if facts["rule_forced"]:
        parts.append("stops forced by a special tyre rule")
    fig.text(0.055, 0.938, "   |   ".join(parts), color=INK_SOFT, fontsize=12, va="center")
    if note:
        fig.text(
            0.975, 0.976, note, color=SURFACE, fontsize=13, fontweight="bold", ha="right", va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": SAFETY_CAR, "edgecolor": "none"},
        )  # fmt: skip
    source = "Timing data: FastF1 (unofficial; not affiliated with Formula 1)."
    if "Source" in race.columns and (race["Source"] == "tracinginsights").any():
        source = (
            "Timing data: FastF1 lap tables from the TracingInsights archive "
            "(unofficial; not affiliated with Formula 1)."
        )
    split = {"train": "training", "valid": "validation", "test": "test", "extra_test": "extra test"}
    source += f"  This race is in the {split.get(facts['split'], facts['split'])} split."
    fig.text(0.055, 0.016, source, color=INK_MUTED, fontsize=10, va="center")

    if out is None:
        out = cfg.results_dir / "figures" / f"race_poster_{race_id}.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--race", help="race id, e.g. 2025_05 (season_round)")
    parser.add_argument("--list", action="store_true", help="list the races that can be drawn")
    parser.add_argument("--model", default="lstm_tyre_aware", choices=sorted(LABELS))
    parser.add_argument("--note", help="banner text, e.g. 'PRACTICE DATA - not real results'")
    parser.add_argument("--out", help="output PNG (default: results/figures/race_poster_<race>.png)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        if args.list or not args.race:
            print(list_races(cfg).to_string(index=False))
            if not args.race:
                print("\nChoose one with --race <RaceID>")
            return
        path = build_poster(cfg, args.race, out=args.out, model=args.model, note=args.note)
    except PosterError as error:
        raise SystemExit(f"race_poster: {error}") from None
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
