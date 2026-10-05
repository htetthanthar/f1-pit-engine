"""F1 tyre-aware pit stop engine: dashboard.

Run:  streamlit run app/streamlit_app.py

The app only reads the exported bundle (see f1pit.dashboard). It never trains or scores a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))  # lets the deployed app run without installing the package

from f1pit import dashboard as dash  # noqa: E402

st.set_page_config(page_title="F1 Pit Stop Engine", page_icon="🏁", layout="wide")

BLUE, RED, GREY, AMBER = "#1f77b4", "#cb181d", "#7f7f7f", "#e6ab02"


@st.cache_resource
def get_bundle(path: str, stamp: float) -> dash.Bundle:
    """`stamp` is the bundle's modification time, so a newly exported bundle replaces the cached one."""
    return dash.load_bundle(Path(path))


def group_label(split: str, subset: str) -> str:
    if split == "valid":
        return "Validation season"
    return f"{dash.SPLIT_LABELS[split]} season: {dash.SUBSET_LABELS[subset].lower()}"


def as_text(table: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Numbers to three decimals, and "n/a" where a value does not apply."""
    out = table.copy()
    for column in numeric:
        out[column] = [("n/a" if pd.isna(v) else f"{v:.3f}") for v in out[column]]
    return out


def percent(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.0%}"


try:
    bundle_dir = dash.find_bundle(APP_DIR)
    bundle = get_bundle(str(bundle_dir), (bundle_dir / "bundle.json").stat().st_mtime)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.title("F1 Tyre-Aware Pit Stop Engine")
st.caption(
    "A two-stage system: Stage 1 forecasts tyre degradation, Stage 2 uses that forecast to predict the lap "
    "on which a driver pits. Results are from seasons the models never saw during training."
)
if bundle.is_demo:
    st.warning(
        "**Demonstration data.** " + (bundle.meta.get("demo_note") or "") + " These are not real Formula 1 "
        "results. Run the pipeline on real data and `python -m f1pit.dashboard` to replace them."
    )

groups = dash.available_groups(bundle)
models = bundle.models
if not groups or not models:
    st.error(
        "The dashboard data holds no results. Run the final evaluation, then `python -m f1pit.dashboard`."
    )
    st.stop()
for notice in dash.notices(bundle):
    st.error(notice)
held_out = [g for g in groups if g[0] != "valid"]
window = dash.pit_window(bundle)
found_column = f"Stop found within ±{window} laps"
default_model = "lstm_tyre_aware" if "lstm_tyre_aware" in models else models[0]

overview, replay, results, hypotheses, explain, about = st.tabs(
    ["Overview", "Race replay", "Model results", "Hypotheses", "Why the models decide", "About"]
)

# ---------------------------------------------------------------------------------------------- overview
with overview:
    split, subset, planned = dash.headline_group(bundle)
    table = dash.model_table(bundle, split, subset)
    info = dash.group_info(bundle, split, subset)
    st.subheader(
        f"{'Headline' if planned else 'Shown'}: {group_label(split, subset)} ({info['n_races']} races)"
    )
    if not planned:
        st.info(
            "The planned headline (the test season, "
            + dash.SUBSET_LABELS[bundle.final["protocol"]["primary_subset"]].lower()
            + ") is not in these results, so this group is shown instead."
        )
    if info["warning"]:
        st.warning(info["warning"].capitalize() + ".")
    main = table[table["key"] == default_model].iloc[0]  # default_model is always one of the models shown
    best = table.sort_values("PR-AUC", ascending=False).iloc[0]
    no_skill = dash.no_skill_level(bundle, split, subset)
    a, b, c, d = st.columns(4)
    a.metric(
        f"PR-AUC, {main['Model']}",
        f"{main['PR-AUC']:.3f}",
        help="Area under the precision-recall curve. 1.0 is perfect.",
    )
    b.metric("PR-AUC of random guessing", f"{no_skill:.3f}", help="The share of laps that are pit laps.")
    c.metric(f"Real stops found within ±{window} laps", percent(main["Stop found within window"]))
    d.metric("Highest PR-AUC", f"{best['PR-AUC']:.3f}", help=f"Model: {best['Model']}")
    d.caption(best["Model"])

    tests = dash.hypothesis_table(bundle)
    mine = tests[
        (tests["Season"] == dash.SPLIT_LABELS[split]) & (tests["Races"] == dash.SUBSET_LABELS[subset])
    ]
    mine = mine[mine["key"].str.startswith("H")]
    if mine.empty:
        st.info("No hypothesis was tested on this group: both models of a hypothesis must be trained.")
    else:
        st.subheader("The hypotheses on this season")
        for _, row in mine.iterrows():
            st.markdown(
                f"**{row['Comparison']}**  \n{row['Reading']}. PR-AUC difference {row['PR-AUC difference']:+.3f} "
                f"(95% CI {row['95% CI low']:+.3f} to {row['95% CI high']:+.3f})."
            )
    st.caption(
        "A hypothesis counts as supported only when the whole 95% interval is above zero. "
        "A result that does not support a hypothesis is reported as it is."
    )

# ------------------------------------------------------------------------------------------------ replay
with replay:
    st.subheader("Race replay: what the pit wall would have seen")
    race_list = dash.races(bundle)
    left, right = st.columns([2, 1])
    race_label = left.selectbox("Race", race_list["Label"], key="race")
    model = right.selectbox(
        "Model",
        models,
        index=models.index(default_model),
        format_func=dash.MODEL_LABELS.get,
        key="replay_model",
    )
    race = race_list[race_list["Label"] == race_label].iloc[0]
    race_id = race["RaceID"]
    if bool(race["RuleForced"]):
        st.info("Stops in this race were forced by a special rule, so it is outside the headline results.")
    lap = st.slider("Lap being driven", 1, int(race["Laps"]), min(15, int(race["Laps"])), key="lap")
    wall = dash.pit_wall(bundle, race_id, lap, model)
    value_name = "Pit probability" if "Pit probability" in wall else "Score"
    calls = wall[wall["Recommendation"] == "PIT THIS LAP"]
    st.markdown(
        f"**Lap {lap}:** the model calls **{len(calls)}** driver(s) in"
        + (": " + ", ".join(calls["Driver"]) if len(calls) else "")
        + ". Everything in this table was known before the lap ended."
    )
    st.dataframe(
        wall,
        hide_index=True,
        width="stretch",
        column_config={
            value_name: st.column_config.ProgressColumn(
                value_name, min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Position": st.column_config.NumberColumn("Position (previous lap)", format="%d"),
        },
    )
    if wall.empty:
        st.info("No driver has a recorded lap with this number.")
    revealed = st.checkbox("Reveal what really happened (this lap, and the rest of the race)", key="reveal")
    if revealed:
        pitted = dash.reveal(bundle, race_id, lap)
        st.success(
            "Pitted at the end of this lap: " + ", ".join(pitted) if pitted else "Nobody pitted on this lap."
        )

    st.divider()
    drivers = sorted(dash.race_laps(bundle, race_id)["Driver"].unique())
    driver = st.selectbox("Driver timeline", drivers, key="driver")
    timeline = dash.driver_timeline(bundle, race_id, driver, model, up_to_lap=None if revealed else lap)
    _, value_label = dash.probability_column(bundle, model)
    lap_axis = alt.X("Lap:Q", title="Lap", scale=alt.Scale(domain=[1, int(race["Laps"])]))
    base = alt.Chart(timeline).encode(x=lap_axis)
    line = base.mark_line(color=BLUE).encode(
        y=alt.Y("Value:Q", title=value_label),
        tooltip=["Lap", alt.Tooltip("Value:Q", format=".3f"), "Tyre", "Tyre age"],
    )
    flagged = (
        base.transform_filter("datum['Model says pit'] == 1")
        .mark_point(color=BLUE, filled=True, size=70)
        .encode(y="Value:Q")
    )
    real = base.transform_filter("datum['Real stop'] == 1").mark_rule(color=RED, strokeDash=[5, 3], size=2)
    safety = (
        base.transform_filter("datum['Safety car'] == 1")
        .mark_tick(color=AMBER, thickness=3, size=14)
        .encode(y=alt.value(0))
    )
    st.altair_chart(alt.layer(line, flagged, real, safety).properties(height=300), width="stretch")
    st.caption(
        "Blue line: the model's value each lap. Blue dots: laps where the model says pit. "
        "Red dashed lines: real pit stops. Amber marks along the top: safety car or virtual safety car laps. "
        + (
            "The whole race is shown."
            if revealed
            else f"Only laps up to lap {lap} are shown, and only stops already made. Tick the box above to see the rest."
        )
    )
    if revealed:
        card = dash.race_scorecard(bundle, race_id, model, window)
        a, b, c = st.columns(3)
        a.metric("Real stops in this race", card["stops"])
        b.metric(f"Found within ±{window} laps", f"{card['found']} of {card['stops']}")
        c.metric("False alarms", f"{card['false_alarms']} of {card['alarms']} calls")

# ----------------------------------------------------------------------------------------------- results
with results:
    choice = st.selectbox(
        "Season and races",
        groups,
        index=groups.index(held_out[0]) if held_out else 0,
        format_func=lambda g: group_label(*g),
        key="group",
    )
    table = dash.model_table(bundle, *choice).rename(columns={"Stop found within window": found_column})
    info = dash.group_info(bundle, *choice)
    st.subheader(f"{group_label(*choice)} ({info['n_races']} races)")
    if info["warning"]:
        st.warning(info["warning"].capitalize() + ".")
    no_skill = dash.no_skill_level(bundle, *choice)
    order = list(table["Model"])
    points = (
        alt.Chart(table)
        .mark_point(filled=True, size=90, color=BLUE)
        .encode(
            x=alt.X("PR-AUC:Q", scale=alt.Scale(domain=[0, 1]), title="PR-AUC with 95% race-level interval"),
            y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
            tooltip=[
                "Model",
                alt.Tooltip("PR-AUC:Q", format=".3f"),
                alt.Tooltip("95% CI low:Q", format=".3f"),
                alt.Tooltip("95% CI high:Q", format=".3f"),
            ],
        )
    )
    bars = (
        alt.Chart(table)
        .mark_rule(color=BLUE, size=2)
        .encode(x="95% CI low:Q", x2="95% CI high:Q", y=alt.Y("Model:N", sort=order))
    )
    guess = (
        alt.Chart(pd.DataFrame({"x": [no_skill]})).mark_rule(color=GREY, strokeDash=[3, 3]).encode(x="x:Q")
    )
    st.altair_chart(alt.layer(bars, points, guess).properties(height=40 * len(table) + 40), width="stretch")
    st.caption(
        "Dotted grey line: random guessing. Intervals resample whole races, because laps in one race are not independent."
    )
    numeric = [c for c in table.columns if c not in ("Model", "key")]
    st.dataframe(
        as_text(table.drop(columns="key"), numeric),
        hide_index=True,
        width="stretch",
    )

# -------------------------------------------------------------------------------------------- hypotheses
with hypotheses:
    tests = dash.hypothesis_table(bundle)
    if tests.empty:
        st.info(
            "No hypothesis tests in these results: both models of a hypothesis must be trained in one Stage 2 run."
        )
    else:
        st.markdown(
            "Each row compares two models on the same laps. The difference is the first model's PR-AUC minus the "
            "second's, with a 95% interval from resampling whole races."
        )
        for key, label in dash.HYPOTHESIS_LABELS.items():
            part = tests[tests["key"] == key].copy()
            if part.empty:
                continue
            st.subheader(label)
            part["Where"] = part["Season"] + ", " + part["Races"].str.lower()
            order = list(part["Where"])
            colour = alt.Color(
                "Reading:N",
                scale=alt.Scale(
                    domain=[
                        "Supported: the whole interval is above zero",
                        "Not supported: the interval includes zero",
                        "Contradicted: the whole interval is below zero",
                        "Too few races to conclude",
                    ],
                    range=[BLUE, GREY, RED, AMBER],
                ),
                legend=alt.Legend(orient="bottom", title=None, labelLimit=400, columns=2),
            )
            chart = alt.Chart(part)
            rule = chart.mark_rule(size=2).encode(
                x=alt.X("95% CI low:Q", title="PR-AUC difference"),
                x2="95% CI high:Q",
                y=alt.Y("Where:N", sort=order, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                color=colour,
            )
            dot = chart.mark_point(filled=True, size=90).encode(
                x="PR-AUC difference:Q",
                y=alt.Y("Where:N", sort=order),
                color=colour,
                tooltip=["Where", alt.Tooltip("PR-AUC difference:Q", format="+.3f"), "Reading"],
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
            st.altair_chart(
                alt.layer(zero, rule, dot).properties(height=45 * len(part) + 70), width="stretch"
            )
            shown = part.drop(columns=["key", "Comparison", "Where"])
            numeric = [
                "PR-AUC difference",
                "95% CI low",
                "95% CI high",
                "Bootstrap p",
                "Holm-adjusted p",
                "McNemar p",
            ]
            st.dataframe(
                as_text(shown, numeric),
                hide_index=True,
                width="stretch",
            )

# ----------------------------------------------------------------------------------------------- explain
with explain:
    splits = [s for s, e in bundle.final["splits"].items() if e.get("explanations")]
    if not splits:
        st.info("No explanations in these results.")
    else:
        season = st.selectbox("Season", splits, format_func=dash.SPLIT_LABELS.get, key="explain_split")
        left, right = st.columns(2)
        shap = dash.importance_table(bundle, season, "xgboost_shap")
        if not shap.empty:
            left.subheader("XGBoost: SHAP values")
            left.altair_chart(
                alt.Chart(shap)
                .mark_bar()
                .encode(
                    x=alt.X("Importance:Q", title="Mean |SHAP value| (log-odds of a pit stop)"),
                    y=alt.Y(
                        "Feature:N",
                        sort=list(shap["Feature"]),
                        title=None,
                        axis=alt.Axis(labelLimit=320, labelOverlap=False),
                    ),
                    color=alt.Color(
                        "Direction:N",
                        scale=alt.Scale(
                            domain=[
                                "Higher value: more likely to pit",
                                "Higher value: less likely to pit",
                                "No clear direction",
                            ],
                            range=[RED, BLUE, GREY],
                        ),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=400, columns=1),
                    ),
                    tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f"), "Direction"],
                )
                .properties(height=24 * len(shap) + 60),
                width="stretch",
            )
        perm = dash.importance_table(bundle, season, "lstm_tyre_aware_permutation")
        if not perm.empty:
            right.subheader("Main LSTM: permutation importance")
            right.altair_chart(
                alt.Chart(perm)
                .mark_bar()
                .encode(
                    x=alt.X("Importance:Q", title="Drop in PR-AUC when the input is shuffled"),
                    y=alt.Y(
                        "Feature:N",
                        sort=list(perm["Feature"]),
                        title=None,
                        axis=alt.Axis(labelLimit=320, labelOverlap=False),
                    ),
                    color=alt.Color(
                        "Direction:N",
                        scale=alt.Scale(
                            domain=["Stage 1 input", "Other input"], range=["#08519c", "#9ecae1"]
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
                )
                .properties(height=24 * len(perm) + 60),
                width="stretch",
            )
            right.caption(
                "Dark bars are the Stage 1 tyre-forecast inputs: how much the main model relies on them."
            )

        st.divider()
        st.subheader("Are the probabilities honest?")
        subset = bundle.final["protocol"]["primary_subset"]
        probability_models = [
            m for m in models if not dash.calibration_table(bundle, season, subset, m).empty
        ]
        if probability_models:
            chosen = st.selectbox(
                "Model",
                probability_models,
                index=probability_models.index(default_model) if default_model in probability_models else 0,
                format_func=dash.MODEL_LABELS.get,
                key="cal_model",
            )
            curve = dash.calibration_table(bundle, season, subset, chosen)
            top = float(max(curve["Predicted probability"].max(), curve["Observed pit rate"].max()))
            diagonal = (
                alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]}))
                .mark_line(color=GREY, strokeDash=[3, 3])
                .encode(
                    x=alt.X("x:Q", title="Predicted probability"), y=alt.Y("y:Q", title="Observed pit rate")
                )
            )
            lines = (
                alt.Chart(curve)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Predicted probability:Q"),
                    y=alt.Y("Observed pit rate:Q"),
                    color=alt.Color(
                        "Version:N",
                        scale=alt.Scale(domain=["As trained", "After Platt scaling"], range=[AMBER, BLUE]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=[
                        "Version",
                        alt.Tooltip("Predicted probability:Q", format=".3f"),
                        alt.Tooltip("Observed pit rate:Q", format=".3f"),
                        "n",
                    ],
                )
            )
            st.altair_chart(alt.layer(diagonal, lines).properties(height=320), width="stretch")
            st.caption(
                "Each point is a tenth of the laps. On the dotted line, a predicted 20% means 20% of those laps were "
                "pit laps. The correction is fitted on held-out training races, never on these seasons."
            )

# ------------------------------------------------------------------------------------------------- about
with about:
    protocol = bundle.final["protocol"]
    history = protocol.get("window_laps", 10)
    listed = protocol.get("seasons") or {
        "train": [2022, 2023],
        "valid": [2024],
        "test": [2025],
        "extra_test": [2026],
    }
    seasons = {
        "train": " and ".join(map(str, listed["train"])),
        "valid": " and ".join(map(str, listed["valid"])),
        "test": " and ".join(map(str, listed["test"] + listed.get("extra_test", []))),
    }
    st.markdown(
        f"""
**Research question.** Does adding an explicit tyre-degradation forecast improve machine-learning prediction of
pit-stop laps?

**How it works.** Stage 1 estimates, for every lap, how many more laps the tyres can run before the
fuel-corrected pace loss passes a threshold. Stage 2 predicts, for every lap, whether the driver pits at the
end of it, from the last {history} laps of race state plus the Stage 1 forecast.

**How the test is kept fair.**
- Models are trained on {seasons["train"]}, compared on {seasons["valid"]}, and tested on {seasons["test"]}.
- Every input for a lap is known before that lap ends; nothing from later laps is used.
- Decision thresholds and probability corrections come from held-out training races, not from the test seasons.
- Wet races are removed. Races whose stops were forced by a special rule are reported separately.

**These results.** Models `{protocol.get("models_fingerprint")}`, evaluation number
{protocol.get("evaluation_number")} of the test seasons. {protocol.get("confidence_intervals", "")}.

**Data.** Timing data comes from the FastF1 library, which reads Formula 1's public live-timing feed. FastF1 is
unofficial and is not associated with Formula 1. This is a university project, not a betting or team tool.
"""
    )
    st.caption(
        f"Dashboard data created {bundle.meta.get('created')}: {bundle.meta.get('races')} races, {bundle.meta.get('laps'):,} laps."
    )
