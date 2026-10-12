"""Tests for dashboard/model_comparison.py (no Streamlit/MLflow server needed).

Run records mimic what ml/train.py logs: validation_* metrics on the model runs
and test_* metrics on the selected_model_test_evaluation run. Values for the
Logistic Regression / Random Forest validation metrics are the project's
reported results; everything else is synthetic fixture data.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import model_comparison as mc  # noqa: E402

MS = 1_700_000_000_000


def rec(run_id, name, status="FINISHED", start=0, metrics=None, params=None):
    return {"run_id": run_id, "run_name": name, "status": status,
            "start_time_ms": MS + start, "end_time_ms": MS + start + 1,
            "artifact_uri": "mlflow-artifacts:/x", "metrics": metrics or {},
            "params": params or {}, "tags": {}}


LR_VAL = {"validation_accuracy": 0.6893293812523157, "validation_precision": 0.6724198661500528,
          "validation_recall": 0.71875, "validation_f1": 0.694813466787989,
          "validation_roc_auc": 0.75}  # roc_auc value is synthetic
RF_VAL = {"validation_accuracy": 0.6963690255650241, "validation_precision": 0.7028320702034304,
          "validation_recall": 0.6634036144578314, "validation_f1": 0.6825489056749952}
# RF has no validation_roc_auc on purpose (missing-metric case)

TEST_METRICS = {"test_accuracy": 0.6775, "test_precision": 0.5989, "test_recall": 0.6872,
                "test_f1": 0.64, "test_roc_auc": 0.7462}


def two_model_frame():
    return mc.runs_to_frame([
        rec("a" * 32, "logistic_regression", start=1, metrics=LR_VAL),
        rec("b" * 32, "random_forest", start=2, metrics=RF_VAL),
        rec("c" * 32, mc.EVALUATION_RUN_NAME, start=3, metrics=TEST_METRICS,
            params={"selected_model": "logistic_regression", "selected_run_id": "a" * 32}),
    ])


def test_classify_run():
    assert mc.classify_run(rec("1", "logistic_regression", metrics=LR_VAL)) == "training"
    assert mc.classify_run(rec("2", mc.EVALUATION_RUN_NAME, metrics=TEST_METRICS)) == "evaluation"
    assert mc.classify_run(rec("3", "scratch", metrics={})) == "other"
    assert mc.classify_run(rec("4", "x", metrics={"test_f1": 0.5})) == "evaluation"


def test_two_models_comparable_and_evaluation_run_excluded():
    frame = two_model_frame()
    selected, notes = mc.select_runs(frame, "latest")
    assert sorted(selected["model_label"]) == ["Logistic Regression", "Random Forest"]
    assert notes == []
    assert mc.available_metrics(selected, "validation") == mc.METRIC_KEYS
    assert mc.available_metrics(selected, "test") == []  # training runs have no test_* metrics


def test_missing_metric_is_nan_not_zero_and_blocks_invalid_winner():
    selected, _ = mc.select_runs(two_model_frame(), "latest")
    table = mc.comparison_table(selected, "validation")
    rf = table[table["Model"] == "Random Forest"].iloc[0]
    assert pd.isna(rf["ROC-AUC"])
    assert mc.fmt_score(rf["ROC-AUC"]) == "—"
    verdict = mc.best_model(selected, "validation", "roc_auc")
    assert verdict["valid"] is False and verdict["winner"] is None
    assert "Random Forest" in verdict["message"]


def test_best_model_f1_matches_reported_results():
    selected, _ = mc.select_runs(two_model_frame(), "latest")
    verdict = mc.best_model(selected, "validation", "f1")
    assert verdict["valid"] and verdict["winner"] == "Logistic Regression"
    verdict = mc.best_model(selected, "validation", "precision")
    assert verdict["winner"] == "Random Forest"  # different criterion, different winner


def test_tie_has_no_winner():
    frame = mc.runs_to_frame([
        rec("a" * 32, "logistic_regression", metrics={"validation_f1": 0.7}),
        rec("b" * 32, "random_forest", metrics={"validation_f1": 0.7}),
    ])
    selected, _ = mc.select_runs(frame, "latest")
    verdict = mc.best_model(selected, "validation", "f1")
    assert verdict["valid"] and verdict["winner"] is None and "Tie" in verdict["message"]


def test_latest_vs_best_policy_with_reruns():
    frame = mc.runs_to_frame([
        rec("a" * 32, "logistic_regression", start=1, metrics={"validation_f1": 0.80}),
        rec("b" * 32, "logistic_regression", start=9, metrics={"validation_f1": 0.60}),
        rec("c" * 32, "random_forest", start=5, metrics={"validation_f1": 0.70}),
    ])
    latest, _ = mc.select_runs(frame, "latest")
    best, _ = mc.select_runs(frame, "best", "f1")
    lr_latest = latest[latest["model"] == "logistic_regression"].iloc[0]
    lr_best = best[best["model"] == "logistic_regression"].iloc[0]
    assert lr_latest["run_id"].startswith("b") and lr_best["run_id"].startswith("a")
    assert int(lr_latest["n_runs"]) == 2


def test_best_policy_excludes_models_without_criterion():
    frame = mc.runs_to_frame([
        rec("a" * 32, "logistic_regression", metrics={"validation_f1": 0.7, "validation_roc_auc": 0.8}),
        rec("b" * 32, "random_forest", metrics={"validation_f1": 0.6}),
    ])
    selected, notes = mc.select_runs(frame, "best", "roc_auc")
    assert list(selected["model_label"]) == ["Logistic Regression"]
    assert any("Random Forest" in n for n in notes)


def test_unfinished_runs_ignored():
    frame = mc.runs_to_frame([
        rec("a" * 32, "logistic_regression", status="FAILED", metrics={"validation_f1": 0.9}),
        rec("b" * 32, "random_forest", status="RUNNING", metrics={"validation_f1": 0.9}),
    ])
    selected, notes = mc.select_runs(frame, "latest")
    assert selected.empty and notes


def test_empty_and_no_metric_runs():
    empty = mc.runs_to_frame([])
    selected, _ = mc.select_runs(empty, "latest")
    assert selected.empty
    assert mc.available_metrics(selected, "validation") == []
    assert mc.best_model(selected, "validation", "f1")["valid"] is False
    no_metrics = mc.runs_to_frame([rec("a" * 32, "logistic_regression")])
    selected, notes = mc.select_runs(no_metrics, "latest")
    assert selected.empty and notes


def test_local_vs_mlflow_differences():
    selected, _ = mc.select_runs(two_model_frame(), "latest")
    same = pd.DataFrame([{"Model": "Logistic Regression", "Accuracy": LR_VAL["validation_accuracy"],
                          "F1-score": LR_VAL["validation_f1"]}])
    assert mc.compare_local_to_mlflow(same, selected) == []
    assert mc.local_vs_mlflow_overlap(same, selected) == 2
    different = same.assign(**{"F1-score": 0.5})
    diffs = mc.compare_local_to_mlflow(different, selected)
    assert len(diffs) == 1 and "F1-score" in diffs[0]
    assert mc.compare_local_to_mlflow(pd.DataFrame(), selected) == []


def test_uri_validation_and_masking():
    assert mc.validate_tracking_uri("http://mlflow:5000") is None
    assert mc.validate_tracking_uri("mlflow:5000") is not None
    assert mc.validate_tracking_uri("") is not None
    assert mc.validate_tracking_uri("sqlite:///mlflow.db") is not None
    assert mc.safe_uri("http://user:secret@host:5000") == "http://***@host:5000"


def test_fetch_rejects_bad_uri_before_touching_mlflow():
    out = mc.fetch_mlflow_runs("localhost:5000", "ecommerce-customer-churn")
    assert out["status"] == "invalid_uri" and out["runs"] == []


def test_fetch_reports_missing_mlflow_package_or_unreachable_server():
    # In an environment without mlflow this is "missing_dependency"; with mlflow
    # installed and nothing listening it must be "unreachable". Never an exception.
    out = mc.fetch_mlflow_runs("http://127.0.0.1:9", "ecommerce-customer-churn")
    assert out["status"] in {"missing_dependency", "unreachable", "api_error"}
    assert out["runs"] == [] and out["message"]


def test_classification_metrics_from_sklearn_report(tmp_path):
    report = {"0": {"precision": 0.7, "recall": 0.66, "f1-score": 0.68, "support": 10},
              "1": {"precision": 0.6724198661500528, "recall": 0.71875,
                    "f1-score": 0.694813466787989, "support": 10},
              "accuracy": 0.6893293812523157}
    path = tmp_path / "logistic_regression_validation_report.json"
    path.write_text(json.dumps(report))
    data, problem = mc.read_json_checked(path)
    assert problem is None
    metrics = mc.classification_metrics(data)
    assert metrics == {"Accuracy": 0.6893293812523157, "Precision": 0.6724198661500528,
                       "Recall": 0.71875, "F1-score": 0.694813466787989}
    assert mc.classification_metrics({}) == {}  # missing keys -> nothing, not zeros


def test_read_json_checked_problems(tmp_path):
    assert mc.read_json_checked(tmp_path / "nope.json")[1] is not None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert "not valid JSON" in mc.read_json_checked(bad)[1]
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2]")
    assert mc.read_json_checked(arr)[0] is None