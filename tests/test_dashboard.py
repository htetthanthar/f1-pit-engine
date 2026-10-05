import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1pit import dashboard as dash

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "app" / "demo_data"


def tiny_bundle() -> dash.Bundle:
    """One race, two drivers, six laps. Driver A stops on lap 3; B never stops."""
    rows = []
    for driver, position in (("A", 1.0), ("B", 2.0)):
        for lap in range(1, 7):
            rows.append(
                {
                    "RaceID": "2025_01",
                    "EventName": "Test Grand Prix",
                    "Driver": driver,
                    "LapNumber": float(lap),
                    "PitLap": int(driver == "A" and lap == 3),
                    "Split": "test",
                    "RuleForced": False,
                    "SafetyCar": int(lap == 5),
                    "TyreLife": float(lap if not (driver == "A" and lap > 3) else lap - 3),
                    "Compound": "SOFT",
                    "Stint": 1.0,
                    "LapsToThreshold": 6 - lap,
                    "Censored": lap == 1,
                    "PrevPosition": position,
                    "m_calibrated": 0.1 * lap,
                    "m_pred": int((driver == "A" and lap in (2, 6)) or (driver == "B" and lap == 4)),
                    "rule_based_score": 0.5,
                    "rule_based_pred": 0,
                }
            )
    final = {"protocol": {"primary_subset": "excluding_rule_forced"}, "splits": {}}
    return dash.Bundle(meta={"demo": False}, final=final, validation=None, laps=pd.DataFrame(rows))


def test_pit_wall_shows_only_what_was_known():
    bundle = tiny_bundle()
    wall = dash.pit_wall(bundle, "2025_01", 4, "m")
    assert wall["Driver"].tolist() == ["A", "B"]  # ordered by position on the previous lap
    assert wall["Stops so far"].tolist() == [1, 0]  # A's stop on lap 3 is in the past
    assert wall["Recommendation"].tolist() == ["Stay out", "PIT THIS LAP"]
    assert wall["Pit probability"].tolist() == pytest.approx([0.4, 0.4])
    assert dash.pit_wall(bundle, "2025_01", 3, "m")["Stops so far"].tolist() == [0, 0]  # not yet known
    assert dash.pit_wall(bundle, "2025_01", 1, "m").iloc[0, 4] == "5+"  # censored forecast is marked

    # changing the outcome of this lap, or anything on later laps, leaves the table unchanged
    later = bundle.laps.copy()
    future = later["LapNumber"] > 4
    later.loc[future, ["PitLap", "m_pred", "SafetyCar"]] = 1
    later.loc[later["LapNumber"] == 4, "PitLap"] = 1
    changed = dash.Bundle(bundle.meta, bundle.final, None, later)
    pd.testing.assert_frame_equal(dash.pit_wall(changed, "2025_01", 4, "m"), wall)

    rule = dash.pit_wall(bundle, "2025_01", 4, "rule_based")
    assert "Score" in rule and "Pit probability" not in rule  # a rule's score is not called a probability


def test_reveal_timeline_and_scorecard():
    bundle = tiny_bundle()
    assert dash.reveal(bundle, "2025_01", 3) == ["A"] and dash.reveal(bundle, "2025_01", 4) == []
    timeline = dash.driver_timeline(bundle, "2025_01", "A", "m")
    assert timeline["Lap"].tolist() == [1, 2, 3, 4, 5, 6] and timeline["Real stop"].sum() == 1
    # while lap 3 is being driven: no later laps, and the stop at the end of lap 3 is not yet known
    live = dash.driver_timeline(bundle, "2025_01", "A", "m", up_to_lap=3)
    assert live["Lap"].tolist() == [1, 2, 3] and live["Real stop"].sum() == 0
    assert dash.driver_timeline(bundle, "2025_01", "A", "m", up_to_lap=4)["Real stop"].tolist() == [
        0,
        0,
        1,
        0,
    ]
    # A: stop on lap 3, calls on laps 2 (within 2 laps: found) and 6 (false alarm). B: one false alarm.
    assert dash.race_scorecard(bundle, "2025_01", "m") == {
        "stops": 1,
        "found": 1,
        "alarms": 3,
        "false_alarms": 2,
    }
    assert dash.race_scorecard(bundle, "2025_01", "rule_based") == {
        "stops": 1,
        "found": 0,
        "alarms": 0,
        "false_alarms": 0,
    }
    listing = dash.races(bundle)
    assert listing["Label"].tolist() == ["2025 Test Grand Prix (round 1)"] and listing["Stops"].tolist() == [
        1
    ]


def test_bundle_without_comparisons_or_optional_columns():
    bundle = tiny_bundle()
    tests = dash.hypothesis_table(bundle)
    assert tests.empty and list(tests.columns) == dash.HYPOTHESIS_COLUMNS  # empty, but still filterable
    assert dash.headline_group(bundle) is None and dash.available_groups(bundle) == []
    assert dash.pit_window(bundle) == 2 and dash.notices(bundle) == []
    bare = dash.Bundle(
        bundle.meta, bundle.final, None, bundle.laps.drop(columns=["LapsToThreshold", "Censored"])
    )
    assert dash.pit_wall(bare, "2025_01", 2, "m").iloc[:, 4].tolist() == ["n/a", "n/a"]
    assert dash.pit_wall(bundle, "2025_01", 99, "m").empty


def test_problems_with_the_results_are_announced():
    bundle = tiny_bundle()
    bundle.final["protocol"]["quick_mode"] = True
    bundle.final["integrity"] = {"status": "not checked"}
    bundle.final["calibrators"] = {"xgboost": {"slope": -1.0, "warning": "slope is not positive"}}
    notes = dash.notices(bundle)
    assert (
        len(notes) == 3 and "quick mode" in notes[0] and "not checked" in notes[1] and "XGBoost" in notes[2]
    )


def test_shap_direction_needs_a_clear_relationship():
    assert dash._direction(0.4) == "Higher value: more likely to pit"
    assert dash._direction(-0.4) == "Higher value: less likely to pit"
    assert dash._direction(0.02) == dash._direction(-0.003) == dash._direction(None) == "No clear direction"


def test_verdict_wording():
    assert dash.verdict({"ci_low": 0.01, "ci_high": 0.05}).startswith("Supported")
    assert dash.verdict({"ci_low": -0.05, "ci_high": -0.01}).startswith("Contradicted")
    assert dash.verdict({"ci_low": -0.01, "ci_high": 0.05}).startswith("Not supported")


def test_demo_bundle_is_complete_and_labelled():
    bundle = dash.load_bundle(DEMO)
    assert bundle.is_demo and "computer-generated" in bundle.meta["demo_note"]
    assert "lstm_tyre_aware" in bundle.models and "rule_based" in bundle.models
    groups = dash.available_groups(bundle)
    assert groups[0] == ("valid", "all_races") and ("test", "excluding_rule_forced") in groups
    assert ("extra_test", "all_races") not in groups  # identical to the headline group, so listed once
    assert dash.headline_group(bundle) == ("test", "excluding_rule_forced", True)
    assert dash.notices(bundle) == []
    for split, subset in groups:
        table = dash.model_table(bundle, split, subset)
        assert len(table) == len(bundle.models) and table["PR-AUC"].between(0, 1).all()
        assert 0 < dash.no_skill_level(bundle, split, subset) < 0.1
    tests = dash.hypothesis_table(bundle)
    assert set(tests["key"]) == set(dash.HYPOTHESIS_LABELS)
    few = tests[tests["Number of races"] < 10]
    assert len(few) and (few["Reading"] == "Too few races to conclude").all()
    assert not dash.importance_table(bundle, "test", "xgboost_shap").empty
    perm = dash.importance_table(bundle, "test", "lstm_tyre_aware_permutation")
    assert (perm["Direction"] == "Stage 1 input").sum() == 3
    curve = dash.calibration_table(bundle, "test", "excluding_rule_forced", "xgboost")
    assert set(curve["Version"]) == {"As trained", "After Platt scaling"}
    assert dash.calibration_table(bundle, "test", "excluding_rule_forced", "rule_based").empty
    race = dash.races(bundle).iloc[0]
    wall = dash.pit_wall(bundle, race["RaceID"], 10, "lstm_tyre_aware")
    assert len(wall) >= 15 and wall["Pit probability"].between(0, 1).all()


def test_bundle_is_found_in_order(tmp_path, monkeypatch):
    monkeypatch.delenv("F1PIT_DASHBOARD_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="python -m f1pit.dashboard"):
        dash.find_bundle(tmp_path)
    for name in ("demo_data", "data", "elsewhere"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "bundle.json").write_text("{}")
        if name == "demo_data":
            assert dash.find_bundle(tmp_path) == tmp_path / "demo_data"
    assert dash.find_bundle(tmp_path) == tmp_path / "data"  # your own results win over the demo
    monkeypatch.setenv("F1PIT_DASHBOARD_DATA", str(tmp_path / "elsewhere"))
    assert dash.find_bundle(tmp_path) == tmp_path / "elsewhere"
    monkeypatch.setenv("F1PIT_DASHBOARD_DATA", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="holds no dashboard data"):  # no silent fallback to the demo
        dash.find_bundle(tmp_path)
    monkeypatch.delenv("F1PIT_DASHBOARD_DATA")
    with pytest.raises(FileNotFoundError, match="not a dashboard bundle"):
        dash.load_bundle(tmp_path / "data")


def test_export_needs_final_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m f1pit.evaluate"):
        dash.export_bundle(tmp_path, tmp_path, tmp_path / "out")


def test_app_runs_and_responds(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("F1PIT_DASHBOARD_DATA", str(DEMO))
    app = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception and not app.error
    assert [t.label for t in app.tabs] == [
        "Overview",
        "Race replay",
        "Model results",
        "Hypotheses",
        "Why the models decide",
        "About",
    ]
    assert any("Demonstration data" in w.value for w in app.warning)
    races = json.loads((DEMO / "bundle.json").read_text())["races"]
    assert len(app.selectbox(key="race").options) == races
    assert np.isfinite(float(app.metric[0].value))

    assert not any(m.label == "Real stops in this race" for m in app.metric)  # hidden until revealed

    app.slider(key="lap").set_value(30)
    app.selectbox(key="replay_model").set_value("rule_based")
    app.checkbox(key="reveal").check()
    app.selectbox(key="group").set_value(("valid", "all_races"))
    app.run()
    assert not app.exception and not app.error
    assert any("Lap 30" in m.value for m in app.markdown)
    assert any(m.label == "Real stops in this race" for m in app.metric)


def test_app_runs_on_a_partial_bundle(tmp_path, monkeypatch):
    """Only two models trained, no hypothesis tests, no validation file, no extra-test season."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    keep = ("rule_based", "xgboost")
    final = json.loads((DEMO / "final_metrics.json").read_text())
    final["splits"]["extra_test"] = {"seasons": [2026], "status": "no races available"}
    for res in final["splits"]["test"]["subsets"].values():
        res["experiments"] = {k: v for k, v in res["experiments"].items() if k in keep}
        res["comparisons"] = {}
    final["splits"]["test"]["explanations"].pop("lstm_tyre_aware_permutation")
    final["protocol"]["quick_mode"] = True
    laps = pd.read_parquet(DEMO / "laps.parquet")
    laps = laps[laps["Split"] == "test"]
    laps = laps[[c for c in laps.columns if not c.startswith(("lstm", "bilstm"))]]
    (tmp_path / "bundle.json").write_text(json.dumps({"demo": False, "races": 13, "laps": len(laps)}))
    (tmp_path / "final_metrics.json").write_text(json.dumps(final))
    laps.to_parquet(tmp_path / "laps.parquet", index=False)

    monkeypatch.setenv("F1PIT_DASHBOARD_DATA", str(tmp_path))
    app = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception
    assert [e.value for e in app.error] == [
        "These models were trained in quick mode, on a few races per season. Not final results."
    ]
    assert app.selectbox(key="replay_model").options == ["Rule-based", "XGBoost"]
    assert any("No hypothesis was tested" in i.value for i in app.info)
