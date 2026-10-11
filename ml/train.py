"""Train/evaluate models and track runs + registered model in MLflow."""
from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml.features import FEATURE_COLUMNS, make_training_data

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "EcommerceChurnClassifier")


def chronological_split(df: pd.DataFrame):
    dates = sorted(df.snapshot_date.unique())
    if len(dates) < 3:
        raise ValueError(f"Need at least 3 distinct snapshot dates for chronological train/validation/test; found {len(dates)}")
    train_end = max(1, int(len(dates) * 0.60))
    val_end = max(train_end + 1, int(len(dates) * 0.80))
    val_end = min(val_end, len(dates) - 1)
    train_dates, val_dates, test_dates = dates[:train_end], dates[train_end:val_end], dates[val_end:]
    return (df[df.snapshot_date.isin(train_dates)].copy(),
            df[df.snapshot_date.isin(val_dates)].copy(),
            df[df.snapshot_date.isin(test_dates)].copy())


def metrics(y_true, y_pred, y_prob):
    out = {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0)),
           "accuracy": float(accuracy_score(y_true, y_pred))}
    try: out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError: out["roc_auc"] = None
    return out


def main():
    data = make_training_data(os.getenv("TRAINING_DATA_PATH"))
    Path("artifacts").mkdir(exist_ok=True)
    data.to_csv("artifacts/customer_training_snapshots.csv", index=False)
    train, val, test = chronological_split(data)
    X_train, y_train = train[FEATURE_COLUMNS], train.is_churned
    X_val, y_val = val[FEATURE_COLUMNS], val.is_churned
    X_test, y_test = test[FEATURE_COLUMNS], test.is_churned
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ecommerce-customer-churn"))
    candidates = {
        "logistic_regression": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                                          ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))]),
                                {"model_type": "LogisticRegression", "max_iter": 1500, "class_weight": "balanced"}),
        "random_forest": (Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                     class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
                          {"model_type": "RandomForest", "n_estimators": 300, "min_samples_leaf": 2,
                           "class_weight": "balanced_subsample"}),
    }
    best = None
    for run_name, (pipeline, params) in candidates.items():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_params({"churn_window_days": 90, "feature_window_days": 180,
                               "split": "chronological by monthly snapshot", "features": ",".join(FEATURE_COLUMNS)})
            pipeline.fit(X_train, y_train)
            val_pred = pipeline.predict(X_val)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            val_metrics = metrics(y_val, val_pred, val_prob)
            mlflow.log_metrics({f"validation_{k}": v for k, v in val_metrics.items() if v is not None})
            report = classification_report(y_val, val_pred, zero_division=0, output_dict=True)
            Path(f"artifacts/{run_name}_validation_report.json").write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(f"artifacts/{run_name}_validation_report.json")
            mlflow.sklearn.log_model(pipeline, artifact_path="model",skops_trusted_types=[
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
],
                                     signature=infer_signature(X_train, pipeline.predict(X_train)))
            candidate = {"name": run_name, "pipeline": pipeline, "run_id": run.info.run_id,
                         "f1": val_metrics["f1"], "params": params}
            if best is None or candidate["f1"] > best["f1"]:
                best = candidate
    # Final test is evaluated only after selecting the candidate on validation F1.
    final_model = best["pipeline"]
    test_pred, test_prob = final_model.predict(X_test), final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, test_pred, test_prob)
    test_report = classification_report(y_test, test_pred, zero_division=0, output_dict=True)
    results = {"selected_model": best["name"], "selected_run_id": best["run_id"],
               "churn_window_days": 90, "feature_window_days": 180,
               "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
               "snapshot_dates": {"train": sorted(train.snapshot_date.unique()), "validation": sorted(val.snapshot_date.unique()), "test": sorted(test.snapshot_date.unique())},
               "test_metrics": test_metrics, "test_classification_report": test_report,
               "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(), "features": FEATURE_COLUMNS}
    Path("artifacts/evaluation_report.json").write_text(json.dumps(results, indent=2))
    Path("artifacts/feature_reference.csv").write_text(train[FEATURE_COLUMNS].to_csv(index=False))
    joblib.dump(final_model, "artifacts/churn_model.joblib")
    client = MlflowClient()
    # Register the selected validation run's model URI; then assign a champion alias.
    model_uri = f"runs:/{best['run_id']}/model"
    try:
        version = mlflow.register_model(model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
        results["registered_model"] = MODEL_NAME
        results["registered_version"] = version.version
        results["registered_alias"] = "champion"
    except Exception as exc:
        results["registry_warning"] = str(exc)
        print(f"WARNING: Model trained but MLflow registration failed: {exc}")
    Path("artifacts/evaluation_report.json").write_text(json.dumps(results, indent=2))
    with mlflow.start_run(run_name="selected_model_test_evaluation"):
        mlflow.log_param("selected_model", best["name"])
        mlflow.log_param("selected_run_id", best["run_id"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if v is not None})
        mlflow.log_artifact("artifacts/evaluation_report.json")
        mlflow.log_artifact("artifacts/feature_reference.csv")
    print(json.dumps(results, indent=2))

if __name__ == "__main__": main()
