import json

import numpy as np
import pandas as pd
import pytest
from synthetic import FakeFastF1

from f1pit.clean import clean
from f1pit.config import load_config
from f1pit.features import build_features, fit_fuel_slopes
from f1pit.ingest import ingest, read_all_races
from f1pit.metrics import (
    apply_platt,
    calibration_table,
    expected_calibration_error,
    fit_platt,
    holm_adjust,
)
from f1pit.stage1 import run_stage1

torch = pytest.importorskip("torch")

from f1pit.evaluate import (  # noqa: E402
    EvalSettings,
    evaluate_and_save,
    load_manifest,
    load_scorers,
    print_summary,
    run_final_evaluation,
    score_part,
    xgboost_shap,
)
from f1pit.stage2 import Stage2Settings, _clean_json, _json_default, prepare, run_stage2  # noqa: E402

CONFIG_TEXT = """
seasons: {train: [2022, 2023], valid: [2024], test: [2025], extra_test: [2026]}
quick: false
quick_races: 4
random_seed: 42
paths: {cache: data/cache, races: data/races, processed: data/processed, results: results}
rule_forced_races: [[2023, Qatar], [2025, Qatar]]
clean: {min_laps_for_fuel_fit: 100, stint_baseline_laps: 3}
stage2:
  window_laps: 5
  seeds: [1, 2]
  max_epochs: 3
  patience: 2
  lstm_hidden: 8
  bilstm_units: [8, 4]
  bilstm_dropout: [0.1, 0.1]
  bilstm_batch_size: 256
  bootstrap_samples: 50
  experiments: [rule_based, xgboost, lstm_tyre_aware, lstm_no_stage1, bilstm_replica]
evaluation: {calibration_bins: 5, permutation_repeats: 1}
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A complete small project on disk: five seasons of fake races, Stage 1, and trained Stage 2 models."""
    root = tmp_path_factory.mktemp("final")
    (root / "configs").mkdir()
    (root / "configs" / "test.yaml").write_text(CONFIG_TEXT)
    cfg = load_config(root / "configs" / "test.yaml", root=root)
    ingest(cfg, fastf1_module=FakeFastF1(rounds=5))
    laps, _ = clean(read_all_races(cfg), cfg)
    slopes = fit_fuel_slopes(laps[(laps["Split"] == "train") & ~laps["RuleForced"]])
    feats, stage1_metrics = run_stage1(build_features(laps, slopes), cfg, model_names=("linear",))
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cfg.processed_dir / "features_stage1.parquet", index=False)
    (cfg.results_dir / "stage1_metrics.json").write_text(json.dumps(stage1_metrics))
    results, predictions = run_stage2(feats, cfg, model_dir=cfg.results_dir / "models")
    (cfg.results_dir / "stage2_metrics.json").write_text(
        json.dumps(_clean_json(results), default=_json_default, allow_nan=False)
    )
    predictions.to_parquet(cfg.results_dir / "stage2_validation_predictions.parquet", index=False)
    return {"cfg": cfg, "feats": feats, "stage2": results, "valid_predictions": predictions}


@pytest.fixture(scope="module")
def final(project):
    cfg = project["cfg"]
    return run_final_evaluation(
        project["feats"],
        cfg,
        cfg.results_dir / "models",
        validation_predictions=project["valid_predictions"],
        stage1_model="linear",
    )


def test_manifest_freezes_the_stage2_thresholds(project):
    manifest = load_manifest(project["cfg"].results_dir / "models")
    assert len(manifest["fingerprint"]) == 16 and manifest["inner_validation_races"]
    assert list(manifest["experiments"]) == list(project["stage2"]["experiments"])  # order is kept
    assert set(manifest["fit_races"]).isdisjoint(manifest["inner_validation_races"])
    for name, e in project["stage2"]["experiments"].items():
        entry = manifest["experiments"][name]
        assert entry["threshold"] == e["seed_ensemble"]["threshold"], name
        assert entry["per_seed_threshold"] == [m["threshold"] for m in e["per_seed"]], name
    assert manifest["experiments"]["rule_based"]["rule_tables"]["by_compound"]
    assert manifest["experiments"]["lstm_tyre_aware"]["files"] == [
        "lstm_tyre_aware_seed1.pt",
        "lstm_tyre_aware_seed2.pt",
    ]


def test_saved_models_reproduce_the_validation_predictions(project):
    cfg, saved = project["cfg"], project["valid_predictions"]
    manifest = load_manifest(cfg.results_dir / "models")
    df = prepare(project["feats"])
    valid = df[df["Split"] == "valid"]
    scorers = {n: load_scorers(n, e, cfg.results_dir / "models") for n, e in manifest["experiments"].items()}
    scored = score_part(valid, scorers, manifest)
    for name, sc in scored.items():
        assert np.allclose(sc["score"], saved[f"{name}_score"], atol=1e-5), name
        assert np.array_equal(sc["pred"], saved[f"{name}_pred"]), name


def test_final_evaluation_covers_both_seasons_and_both_groups_of_races(final):
    results, predictions = final
    assert results["integrity"]["status"] == "ok"
    assert set(predictions["Split"]) == {"test", "extra_test"}
    test = results["splits"]["test"]
    assert test["rule_forced_races"] == ["2025_04"]
    headline, everything = test["subsets"]["excluding_rule_forced"], test["subsets"]["all_races"]
    assert "2025_04" not in headline["races"] and "2025_04" in everything["races"]
    assert everything["n_races"] == headline["n_races"] + 1 and everything["n_laps"] > headline["n_laps"]
    extra = results["splits"]["extra_test"]["subsets"]
    assert (
        "note" in extra["all_races"]
        and extra["all_races"]["n_laps"] == extra["excluding_rule_forced"]["n_laps"]
    )
    for split in ("test", "extra_test"):
        res = results["splits"][split]["subsets"]["excluding_rule_forced"]
        assert set(res["comparisons"]) == {
            "H1_stage1_forecast_helps",
            "H2_sequence_beats_trees",
            "main_model_vs_published_bilstm",
        }
        for label in ("H1_stage1_forecast_helps", "H2_sequence_beats_trees"):
            c = res["comparisons"][label]
            assert c["bootstrap_p_holm"] >= c["pr_auc_difference"]["p_not_better"]
        assert "bootstrap_p_holm" not in res["comparisons"]["main_model_vs_published_bilstm"]
        for name, m in res["experiments"].items():
            assert 0 <= m["pr_auc"] <= 1 and m["pr_auc_ci"]["ci_low"] <= m["pr_auc_ci"]["ci_high"], name
            stops = m["recall_by_track_status"]
            assert stops["safety_car"]["stops"] + stops["green_flag"]["stops"] == m["n_pit_laps"]
        assert res["experiments"]["xgboost"]["pr_auc"] > 3 * res["experiments"]["xgboost"]["no_skill_pr_auc"]
        assert res["stage1"]["mae_5lap_s"] < res["stage1"]["persistence_mae_5lap_s"]


def test_decisions_use_the_frozen_thresholds(project, final):
    _, predictions = final
    manifest = load_manifest(project["cfg"].results_dir / "models")
    for name, entry in manifest["experiments"].items():
        if entry["is_probability"]:
            expected = (predictions[f"{name}_score"] >= entry["threshold"]).astype(int)
            assert np.array_equal(predictions[f"{name}_pred"], expected), name


def test_test_labels_cannot_change_any_prediction(project, final):
    """Flip every test label: scores, decisions and calibrated probabilities must be unchanged."""
    cfg, feats = project["cfg"], project["feats"].copy()
    held_out = feats["Split"].isin(["test", "extra_test"])
    feats.loc[held_out, "PitLap"] = 1 - feats.loc[held_out, "PitLap"]
    _, flipped = run_final_evaluation(feats, cfg, cfg.results_dir / "models")
    _, original = final
    columns = [c for c in original.columns if c.endswith(("_score", "_pred", "_calibrated"))]
    assert columns
    pd.testing.assert_frame_equal(flipped[columns], original[columns])


def test_one_season_does_not_affect_the_other(project, final):
    cfg, feats = project["cfg"], project["feats"]
    results, _ = run_final_evaluation(feats[feats["Split"] != "extra_test"], cfg, cfg.results_dir / "models")
    assert results["splits"]["extra_test"]["status"] == "no races available"
    a = results["splits"]["test"]["subsets"]["excluding_rule_forced"]["experiments"]
    b = final[0]["splits"]["test"]["subsets"]["excluding_rule_forced"]["experiments"]
    for name in a:
        assert a[name]["pr_auc"] == b[name]["pr_auc"] and a[name]["f1"] == b[name]["f1"], name


def test_stale_models_are_refused(project):
    cfg = project["cfg"]
    tampered = project["valid_predictions"].copy()
    tampered["xgboost_score"] = tampered["xgboost_score"] + 0.05
    with pytest.raises(RuntimeError, match="do not reproduce"):
        run_final_evaluation(
            project["feats"], cfg, cfg.results_dir / "models", validation_predictions=tampered
        )
    flipped = project["valid_predictions"].copy()
    flipped.loc[0, "lstm_tyre_aware_pred"] = 1 - flipped.loc[0, "lstm_tyre_aware_pred"]
    with pytest.raises(RuntimeError, match="different validation decisions"):
        run_final_evaluation(
            project["feats"], cfg, cfg.results_dir / "models", validation_predictions=flipped
        )
    with pytest.raises(RuntimeError, match="do not cover 'xgboost'"):
        run_final_evaluation(
            project["feats"],
            cfg,
            cfg.results_dir / "models",
            validation_predictions=project["valid_predictions"].drop(columns="xgboost_score"),
        )
    feats = project["feats"]
    inner = load_manifest(cfg.results_dir / "models")["inner_validation_races"][0]
    with pytest.raises(RuntimeError, match="training races have changed"):
        run_final_evaluation(feats[feats["RaceID"] != inner], cfg, cfg.results_dir / "models")


def test_calibration_and_explanations(project, final):
    results, predictions = final
    res = results["splits"]["test"]["subsets"]["excluding_rule_forced"]
    cal = res["experiments"]["xgboost"]["calibration"]
    assert len(cal["curve"]) == 5 and sum(row["n"] for row in cal["curve"]) == res["n_laps"]
    assert 0 <= cal["ece_after_platt"] <= 1 and 0 <= cal["brier_after_platt"] <= 1
    assert "calibration" not in res["experiments"]["rule_based"]
    assert predictions["xgboost_calibrated"].between(0, 1).all()
    # Platt scaling never changes the order of laps
    by_raw = predictions.sort_values("xgboost_score", kind="stable")
    assert results["calibrators"]["xgboost"]["slope"] > 0
    assert (np.diff(by_raw["xgboost_calibrated"].to_numpy()) >= 0).all()
    assert by_raw["xgboost_calibrated"].nunique() == by_raw["xgboost_score"].nunique()

    expl = results["splits"]["test"]["explanations"]
    shap = expl["xgboost_shap"]
    assert set(shap) == set(
        load_manifest(project["cfg"].results_dir / "models")["experiments"]["xgboost"]["features"]
    )
    assert list(shap)[0] in ("TyreLife", "LapsToThreshold", "LapsRemaining", "RaceProgress", "SafetyCar")
    perm = expl["lstm_tyre_aware_permutation"]["features"]
    assert "Stage 1 forecast (both inputs)" in perm and "LapsToThreshold" in perm
    assert "LapsToThreshold" not in expl["lstm_no_stage1_permutation"]["features"]


def test_shap_values_add_up_to_the_model_output(project):
    import xgboost as xgb

    cfg = project["cfg"]
    entry = load_manifest(cfg.results_dir / "models")["experiments"]["xgboost"]
    df = prepare(project["feats"])
    part = df[df["Split"] == "test"].head(200)
    booster = xgb.Booster()
    booster.load_model(cfg.results_dir / "models" / entry["files"][0])
    data = xgb.DMatrix(part[entry["features"]].astype(float))
    contributions = booster.predict(data, pred_contribs=True)
    assert np.allclose(contributions.sum(axis=1), booster.predict(data, output_margin=True), atol=1e-3)
    assert all(v["mean_abs"] >= 0 for v in xgboost_shap(entry, part, cfg.results_dir / "models").values())


def test_evaluate_and_save_writes_everything_and_runs_once(project):
    cfg = project["cfg"]
    out = cfg.results_dir
    log_path = out / "final_evaluation_log.json"
    results, status = evaluate_and_save(cfg, make_figures=False)
    assert status == "evaluated" and results["protocol"]["evaluation_number"] == 1
    json.loads((out / "final_metrics.json").read_text())  # valid JSON, no NaN
    table = pd.read_csv(out / "final_model_table.csv")
    assert set(table["split"]) == {"valid", "test", "extra_test"}
    hypotheses = pd.read_csv(out / "final_hypothesis_table.csv")
    assert {"H1_stage1_forecast_helps", "H2_sequence_beats_trees"} <= set(hypotheses["comparison"])
    assert len(pd.read_parquet(out / "final_predictions.parquet")) > 0
    assert not (out / "figures").exists()

    # same models, same laps: nothing is evaluated again, but missing charts are drawn from the saved results
    again, status = evaluate_and_save(cfg)
    assert status == "skipped" and again["protocol"]["evaluation_number"] == 1
    assert len(json.loads(log_path.read_text())) == 1
    figures = {p.name for p in (out / "figures").iterdir() if p.stat().st_size > 5000}
    assert {
        "model_comparison.png",
        "hypothesis_tests.png",
        "pr_curves_test.png",
        "calibration_test.png",
        "shap_xgboost_test.png",
        "permutation_lstm_test.png",
        "example_race_test.png",
        "pr_curves_extra_test.png",
    } <= figures

    forced, status = evaluate_and_save(cfg, rerun=True, make_figures=False)
    assert status == "evaluated" and forced["protocol"]["evaluation_number"] == 2
    log = json.loads(log_path.read_text())
    assert len(log) == 2 and log[1]["rule_forced_races"]["test"] == ["2025_04"]

    # a new race in a held-out season is new data: it is evaluated (and logged) without --rerun
    features_path = cfg.processed_dir / "features_stage1.parquet"
    feats = pd.read_parquet(features_path)
    try:
        feats[feats["RaceID"] != "2026_05"].to_parquet(features_path, index=False)
        fewer, status = evaluate_and_save(cfg, make_figures=False)
        assert status == "evaluated" and fewer["protocol"]["evaluation_number"] == 3
        assert fewer["splits"]["extra_test"]["subsets"]["all_races"]["n_races"] == 4
    finally:
        feats.to_parquet(features_path, index=False)


def test_dashboard_bundle_is_exported_from_the_final_results(project, tmp_path):
    from f1pit import dashboard as dash

    cfg = project["cfg"]
    if not (cfg.results_dir / "final_metrics.json").exists():
        evaluate_and_save(cfg, make_figures=False)
    meta = dash.export_bundle(cfg.results_dir, cfg.processed_dir, tmp_path / "bundle")
    bundle = dash.load_bundle(tmp_path / "bundle")
    assert not bundle.is_demo and meta["laps"] == len(bundle.laps)
    assert bundle.models == ["rule_based", "xgboost", "lstm_tyre_aware", "lstm_no_stage1", "bilstm_replica"]
    assert {"LapsToThreshold", "Censored", "PrevPosition", "Stint"} <= set(bundle.laps.columns)
    assert "xgboost_calibrated" in bundle.laps and "xgboost_score" not in bundle.laps  # not needed twice
    assert "rule_based_score" in bundle.laps  # the rule has no probability, so its score is kept
    saved = pd.read_parquet(cfg.results_dir / "final_predictions.parquet")
    merged = saved.merge(bundle.laps, on=["RaceID", "Driver", "LapNumber"], suffixes=("", "_bundle"))
    assert (
        len(merged) == len(saved)
        and (merged["lstm_tyre_aware_pred"] == merged["lstm_tyre_aware_pred_bundle"]).all()
    )
    table = dash.model_table(bundle, "test", "excluding_rule_forced")
    final = json.loads((cfg.results_dir / "final_metrics.json").read_text())
    expected = final["splits"]["test"]["subsets"]["excluding_rule_forced"]["experiments"]["xgboost"]["pr_auc"]
    assert (
        table.loc[table["key"] == "xgboost", "PR-AUC"].iloc[0] == expected
    )  # the app shows the saved number
    wall = dash.pit_wall(bundle, "2025_01", 12, "lstm_tyre_aware")
    assert len(wall) == 20 and wall["Position"].notna().all()
    marked = dash.export_bundle(
        cfg.results_dir, cfg.processed_dir, tmp_path / "demo", demo_note="practice data"
    )
    assert marked["demo"] and dash.load_bundle(tmp_path / "demo").is_demo
    assert dash.pit_window(bundle) == 2 and bundle.final["protocol"]["seasons"]["extra_test"] == [2026]

    # feature files that no longer match the evaluation are refused, not silently joined
    stale = tmp_path / "processed"
    stale.mkdir()
    feats = pd.read_parquet(cfg.processed_dir / "features_stage1.parquet")
    feats[feats["RaceID"] != "2025_02"].to_parquet(stale / "features_stage1.parquet", index=False)
    with pytest.raises(RuntimeError, match="feature files changed"):
        dash.export_bundle(cfg.results_dir, stale, tmp_path / "stale")


def test_evaluation_refuses_quick_models_and_missing_checks(project):
    cfg = project["cfg"]
    out = cfg.results_dir
    manifest_path = out / "models" / "manifest.json"
    original = manifest_path.read_text()
    log_before = (out / "final_evaluation_log.json").read_text()
    try:
        manifest_path.write_text(json.dumps({**json.loads(original), "quick": True}))
        with pytest.raises(RuntimeError, match="quick mode"):
            evaluate_and_save(cfg, rerun=True, make_figures=False)
    finally:
        manifest_path.write_text(original)
    valid_path = out / "stage2_validation_predictions.parquet"
    moved = valid_path.with_suffix(".moved")
    valid_path.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match="check that the saved models"):
            evaluate_and_save(cfg, rerun=True, make_figures=False)
    finally:
        moved.rename(valid_path)
    assert (out / "final_evaluation_log.json").read_text() == log_before  # refused runs are not looks


def test_partial_stage2_run_is_evaluated_and_says_what_is_missing(project, tmp_path, capsys):
    cfg, feats = project["cfg"], project["feats"]
    settings = Stage2Settings.from_dict({**cfg.stage2, "experiments": ["rule_based", "xgboost"]})
    _, valid_predictions = run_stage2(feats, cfg, settings, model_dir=tmp_path / "models")
    results, predictions = run_final_evaluation(
        feats, cfg, tmp_path / "models", validation_predictions=valid_predictions
    )
    res = results["splits"]["test"]["subsets"]["excluding_rule_forced"]
    assert set(res["experiments"]) == {"rule_based", "xgboost"} and res["comparisons"] == {}
    assert res["hypotheses_not_tested"] == ["H1_stage1_forecast_helps", "H2_sequence_beats_trees"]
    print_summary(results)
    assert "H1_stage1_forecast_helps: NOT TESTED" in capsys.readouterr().out
    from f1pit.plots import make_all_figures

    made = {p.name for p in make_all_figures(results, predictions, None, tmp_path / "figures")}
    assert "pr_curves_test.png" in made and "hypothesis_tests.png" not in made


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    table = calibration_table(y, p, bins=10)
    assert len(table) == 10 and sum(r["n"] for r in table) == 20000
    assert expected_calibration_error(table) < 0.02  # well calibrated by construction
    overconfident = np.clip(p * 3, 0, 1)
    assert expected_calibration_error(calibration_table(y, overconfident, 10)) > 0.1
    # Platt scaling repairs a distorted but correctly ordered score
    logit = np.log(p / (1 - p))
    distorted = 1 / (1 + np.exp(-(2.5 * logit + 1.0)))
    params = fit_platt(y, distorted)
    assert params["slope"] == pytest.approx(1 / 2.5, rel=0.1)
    assert expected_calibration_error(calibration_table(y, apply_platt(params, distorted), 10)) < 0.02


def test_holm_adjustment():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm_adjust({"a": 0.6, "b": 0.9}) == pytest.approx({"a": 1.0, "b": 1.0})
    assert holm_adjust({}) == {}


def test_evaluation_settings_are_checked():
    with pytest.raises(ValueError, match="Unknown evaluation settings"):
        EvalSettings.from_dict({"bins": 5})
    with pytest.raises(ValueError, match="primary_subset"):
        EvalSettings.from_dict({"primary_subset": "best_looking"})
    assert Stage2Settings.from_dict({}).window_laps == 10


def test_report_pack_copies_the_saved_numbers(project):
    from f1pit import report

    cfg = project["cfg"]
    if not (cfg.results_dir / "final_metrics.json").exists():
        evaluate_and_save(cfg, make_figures=False)
    pack = report.build_report(cfg)
    assert set(pack) == {"data_facts.md", "results_tables.md"}
    tables, facts = pack["results_tables.md"], pack["data_facts.md"]
    for n in range(2, 10):
        assert f"## Table {n}." in tables, n
    assert "## Table 1." in facts and "QUICK MODE" not in facts

    final = json.loads((cfg.results_dir / "final_metrics.json").read_text())
    head = final["splits"]["test"]["subsets"]["excluding_rule_forced"]
    m = head["experiments"]["xgboost"]
    ci = m["pr_auc_ci"]
    row = f"| XGBoost | {m['pr_auc']:.3f} | {ci['ci_low']:.3f} to {ci['ci_high']:.3f} |"
    section = tables.split("## Table 4.")[1].split("## Table 5.")[0]
    assert row in section and "headline" in tables.split("## Table 4.")[1].splitlines()[0]
    assert f"{head['n_races']} races." in section and "WARNING: only" in section  # 3 races in this fixture
    h1 = head["comparisons"]["H1_stage1_forecast_helps"]["pr_auc_difference"]
    assert f"| {h1['estimate']:+.3f} | {h1['ci_low']:+.3f} to {h1['ci_high']:+.3f} |" in tables
    assert "too few races to conclude" in tables  # no verdict is drawn from a handful of races

    feats = project["feats"]
    test = feats[feats["Split"] == "test"]
    expected = (
        f"| Test | 2025 | {test['RaceID'].nunique()} | 1 | {len(test):,} | {int(test['PitLap'].sum()):,} |"
    )
    assert expected in facts
    assert (
        "Rule-forced races: 2023_04, 2025_04" in facts
        and "Stage 1 model selected on validation: linear" in facts
    )
    log = json.loads((cfg.results_dir / "final_evaluation_log.json").read_text())
    assert f"Evaluations of the test seasons recorded in the log: {len(log)}" in facts


def test_report_formatting_helpers(tmp_path, cfg):
    from f1pit import report

    assert report.num(-0.0004) == "0.000" and report.num(-0.0004, signed=True) == "+0.000"  # never "-0.000"
    assert (
        report.num(None) == "n/a" and report.num(float("nan")) == "n/a" and report.num(0.12345, 2) == "0.12"
    )
    assert (
        report.p_value(0.0004) == "<0.001"
        and report.p_value(0.0312) == "0.031"
        and report.p_value(None) == "n/a"
    )
    above, below, across = (
        {"ci_low": a, "ci_high": b} for a, b in ((0.01, 0.05), (-0.05, -0.01), (-0.01, 0.05))
    )
    assert report.reading(above).startswith("supported") and report.reading(below).startswith("contradicted")
    assert report.reading(across).startswith("not supported")
    assert report.reading(below, is_hypothesis=False).startswith("second model better")
    assert report.reading(across, is_hypothesis=False).startswith("no difference shown")
    assert report.table(["a", "b"], [[1, 2]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"
    with pytest.raises(FileNotFoundError, match="python -m f1pit.evaluate"):
        report.build_report(cfg)
