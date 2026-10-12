"""Helpers for the dashboard's MLflow model comparison section.

Everything here except ``fetch_mlflow_runs`` is plain pandas/Python so it can be
unit-tested without Streamlit, Plotly or a running MLflow server.

Metric names come from ``ml/train.py``:
  * model runs (run names ``logistic_regression`` / ``random_forest``) log
    ``validation_{accuracy,precision,recall,f1,roc_auc}``;
  * the ``selected_model_test_evaluation`` run logs ``test_{...}`` for ONE model
    (the one chosen on validation F1) and params ``selected_model`` and
    ``selected_run_id``.
Validation and test metrics are therefore never mixed in one comparison.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

DEFAULT_EXPERIMENT_NAME = "ecommerce-customer-churn"
EVALUATION_RUN_NAME = "selected_model_test_evaluation"

# Order used everywhere metrics are listed. Keys match ml/train.py.
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}
SPLIT_PREFIX = {"validation": "validation_", "test": "test_"}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Local JSON report label -> MLflow metric key (class "1" = churn).
LOCAL_LABEL_TO_KEY = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-score": "f1",
}

FLOAT_TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# Local JSON reports
# --------------------------------------------------------------------------

def read_json_checked(path: Path) -> tuple[dict | None, str | None]:
    """Return (data, problem). ``problem`` explains a missing/invalid file."""
    if not path.exists():
        return None, f"{path.name} not found in {path.parent}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        return None, f"{path.name} could not be read: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path.name} does not contain a JSON object"
    return data, None


def classification_metrics(report: dict) -> dict[str, float]:
    """Accuracy plus churn-class (label "1") precision/recall/F1.

    Reads sklearn ``classification_report(output_dict=True)`` JSON, the format
    written by ml/train.py. Missing keys are simply omitted (never zero-filled).
    """
    positive = report.get("1", {})
    result: dict[str, float] = {}
    if "accuracy" in report:
        result["Accuracy"] = report["accuracy"]
    if isinstance(positive, dict):
        for label, key in [("Precision", "precision"), ("Recall", "recall"),
                           ("F1-score", "f1-score")]:
            if key in positive:
                result[label] = positive[key]
    return result


# --------------------------------------------------------------------------
# Tracking URI handling
# --------------------------------------------------------------------------

def safe_uri(uri: str) -> str:
    """Hide any ``user:password@`` part before showing a URI."""
    return re.sub(r"://[^/@]*@", "://***@", uri or "")


def validate_tracking_uri(uri: str | None) -> str | None:
    """Return a problem description, or None if the URI is an http(s) URL.

    A non-http value would make the MLflow client silently use a local
    file store (and create an ``mlruns`` folder / a second experiment), so it
    is rejected before the client is built.
    """
    if not uri or not uri.strip():
        return "MLFLOW_TRACKING_URI is empty."
    parsed = urlparse(uri.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return (f"MLFLOW_TRACKING_URI '{safe_uri(uri)}' is not an http(s) URL. "
                "Inside Docker Compose it should be http://mlflow:5000.")
    return None


# --------------------------------------------------------------------------
# Fetching from MLflow (the only function that needs the mlflow package)
# --------------------------------------------------------------------------

def _result(status: str, message: str = "", **extra: Any) -> dict:
    out = {"status": status, "message": message, "runs": [],
           "experiment_id": None,
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.update(extra)
    return out


def fetch_mlflow_runs(tracking_uri: str, experiment_name: str,
                      max_results: int = 500) -> dict:
    """Read runs from an EXISTING experiment. Never creates an experiment.

    Returns a plain dict (cache-friendly) with ``status`` one of:
    ok, invalid_uri, missing_dependency, unreachable, api_error, no_experiment.
    """
    problem = validate_tracking_uri(tracking_uri)
    if problem:
        return _result("invalid_uri", problem)

    # Fail fast instead of MLflow's default long retry/backoff schedule.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

    try:
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        return _result("missing_dependency",
                       f"The mlflow package could not be imported: {exc}")

    shown = safe_uri(tracking_uri)
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return _result(
                "no_experiment",
                f"Experiment '{experiment_name}' does not exist on {shown}. "
                "Run `python -m ml.train` against this server first.")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["attributes.start_time DESC"],
        )
    except MlflowException as exc:
        text = str(exc)
        unreachable = any(token in text for token in
                          ("Max retries", "Connection", "timed out", "Name or service"))
        return _result(
            "unreachable" if unreachable else "api_error",
            f"{type(exc).__name__} while talking to {shown}: "
            f"{safe_uri(text)[:500]}")

    records = []
    for run in runs:
        info, data = run.info, run.data
        records.append({
            "run_id": info.run_id,
            "run_name": data.tags.get("mlflow.runName") or "",
            "status": info.status,
            "start_time_ms": info.start_time,
            "end_time_ms": info.end_time,
            "artifact_uri": info.artifact_uri,
            "metrics": dict(data.metrics),
            "params": dict(data.params),
            "tags": {k: v for k, v in data.tags.items()
                     if not k.startswith("mlflow.")},
        })
    return _result("ok", experiment_id=experiment.experiment_id, runs=records)


# --------------------------------------------------------------------------
# Run classification and frame building
# --------------------------------------------------------------------------

def classify_run(record: dict) -> str:
    """'training' (logs validation_*), 'evaluation' (test-only) or 'other'."""
    metrics = record.get("metrics") or {}
    has_validation = any(k.startswith(SPLIT_PREFIX["validation"]) for k in metrics)
    has_test = any(k.startswith(SPLIT_PREFIX["test"]) for k in metrics)
    if record.get("run_name") == EVALUATION_RUN_NAME or (has_test and not has_validation):
        return "evaluation"
    if has_validation:
        return "training"
    return "other"


def model_key(run_name: str) -> str:
    return re.sub(r"[\s\-]+", "_", (run_name or "").strip().lower()) or "unnamed"


def model_display_name(key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def runs_to_frame(records: list[dict]) -> pd.DataFrame:
    """One row per run. Metric columns keep their logged names."""
    base_cols = ["run_id", "short_id", "run_name", "model", "model_label", "run_type",
                 "status", "start_time", "params", "tags", "artifact_uri"]
    if not records:
        return pd.DataFrame(columns=base_cols)
    rows = []
    for rec in records:
        row = {
            "run_id": rec["run_id"],
            "short_id": rec["run_id"][:8],
            "run_name": rec.get("run_name", ""),
            "model": model_key(rec.get("run_name", "")),
            "run_type": classify_run(rec),
            "status": str(rec.get("status", "")).upper(),
            "start_time": (pd.to_datetime(rec["start_time_ms"], unit="ms", utc=True)
                           if rec.get("start_time_ms") else pd.NaT),
            "params": rec.get("params", {}),
            "tags": rec.get("tags", {}),
            "artifact_uri": rec.get("artifact_uri", ""),
        }
        row["model_label"] = model_display_name(row["model"])
        for key, value in (rec.get("metrics") or {}).items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def metric_column(split: str, metric: str) -> str:
    return f"{SPLIT_PREFIX[split]}{metric}"


def available_metrics(frame: pd.DataFrame, split: str) -> list[str]:
    """Metric keys that at least one run in ``frame`` actually logged for ``split``."""
    out = []
    for metric in METRIC_KEYS:
        col = metric_column(split, metric)
        if col in frame.columns and frame[col].notna().any():
            out.append(metric)
    return out


# --------------------------------------------------------------------------
# Run selection, tables, best model
# --------------------------------------------------------------------------

def select_runs(runs: pd.DataFrame, policy: str = "latest",
                criterion: str = "f1") -> tuple[pd.DataFrame, list[str]]:
    """Pick ONE finished validation-logging training run per model.

    policy ``latest``: newest finished run of each model.
    policy ``best``: the finished run with the highest validation ``criterion``
    (models that never logged the criterion are left out and reported in notes).
    Returns (selected rows, notes). Row count per model is in ``n_runs``.
    """
    notes: list[str] = []
    cols = list(runs.columns) + ["n_runs"]
    if runs.empty:
        return pd.DataFrame(columns=cols), notes

    pool = runs[(runs["run_type"] == "training") & (runs["status"] == "FINISHED")].copy()
    if pool.empty:
        return pd.DataFrame(columns=cols), ["No finished training runs with validation metrics."]

    counts = pool.groupby("model").size()
    pool["n_runs"] = pool["model"].map(counts)
    pool = pool.sort_values("start_time", ascending=False, na_position="last")

    if policy == "best":
        col = metric_column("validation", criterion)
        if col not in pool.columns:
            notes.append(f"No run logged {col}.")
            return pool.iloc[0:0], notes
        have = pool[pool[col].notna()]
        for model in sorted(set(pool["model"]) - set(have["model"])):
            notes.append(f"{model_display_name(model)} has no logged {col}; excluded.")
        selected = (have.sort_values([col, "start_time"], ascending=[False, False],
                                     na_position="last")
                    .drop_duplicates("model", keep="first"))
    else:
        selected = pool.drop_duplicates("model", keep="first")

    return selected.sort_values("model").reset_index(drop=True), notes


def fmt_score(value: Any) -> str:
    """Three decimals, or an em dash for missing values (never 0)."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def comparison_table(selected: pd.DataFrame, split: str,
                     metrics: list[str] | None = None) -> pd.DataFrame:
    """Numeric table: one row per model with the requested metrics (NaN if missing)."""
    metrics = metrics or METRIC_KEYS
    rows = []
    for _, run in selected.iterrows():
        row = {"Model": run["model_label"], "Run ID": run["short_id"],
               "Run started": run["start_time"], "Runs available": int(run.get("n_runs", 1))}
        for metric in metrics:
            col = metric_column(split, metric)
            row[METRIC_LABELS[metric]] = run[col] if col in selected.columns else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def best_model(selected: pd.DataFrame, split: str, criterion: str) -> dict:
    """Winner on ``criterion`` only if >= 2 models logged it for this split."""
    col = metric_column(split, criterion)
    label = METRIC_LABELS.get(criterion, criterion)
    if selected.empty or col not in selected.columns:
        return {"valid": False, "winner": None, "score": None,
                "message": f"No selected run logged {label}."}
    scored = selected[selected[col].notna()]
    if len(scored) < 2:
        missing = sorted(set(selected["model_label"]) - set(scored["model_label"]))
        detail = f" (missing for: {', '.join(missing)})" if missing else ""
        return {"valid": False, "winner": None, "score": None,
                "message": f"Need {label} from at least two models to compare{detail}."}
    top = scored[col].max()
    leaders = scored[(scored[col] - top).abs() <= FLOAT_TOLERANCE]["model_label"].tolist()
    if len(leaders) > 1:
        return {"valid": True, "winner": None, "score": float(top),
                "message": f"Tie on {label} ({top:.3f}): {', '.join(leaders)}."}
    return {"valid": True, "winner": leaders[0], "score": float(top),
            "message": f"{leaders[0]} has the highest validation {label} ({top:.3f})."}


def compare_local_to_mlflow(local: pd.DataFrame, selected: pd.DataFrame) -> list[str]:
    """Describe differences between local validation reports and selected MLflow runs."""
    if local is None or local.empty or selected.empty:
        return []
    diffs = []
    by_label = selected.set_index("model_label")
    for _, row in local.iterrows():
        model = row.get("Model")
        if model not in by_label.index:
            continue
        run = by_label.loc[model]
        for label, key in LOCAL_LABEL_TO_KEY.items():
            col = metric_column("validation", key)
            if label in local.columns and col in by_label.columns \
                    and pd.notna(row[label]) and pd.notna(run[col]) \
                    and abs(float(row[label]) - float(run[col])) > FLOAT_TOLERANCE:
                diffs.append(f"{model} {label}: local report {float(row[label]):.3f} "
                             f"vs MLflow {float(run[col]):.3f}")
    return diffs


def local_vs_mlflow_overlap(local: pd.DataFrame, selected: pd.DataFrame) -> int:
    """How many (model, metric) pairs exist in BOTH sources and were compared."""
    if local is None or local.empty or selected.empty:
        return 0
    by_label = selected.set_index("model_label")
    count = 0
    for _, row in local.iterrows():
        if row.get("Model") not in by_label.index:
            continue
        run = by_label.loc[row["Model"]]
        for label, key in LOCAL_LABEL_TO_KEY.items():
            col = metric_column("validation", key)
            if label in local.columns and col in by_label.columns \
                    and pd.notna(row[label]) and pd.notna(run[col]):
                count += 1
    return count