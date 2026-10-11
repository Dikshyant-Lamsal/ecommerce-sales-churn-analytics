"""Evaluate precision/recall when verified ground-truth churn labels are available."""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="CSV containing is_churned and churn_prediction columns")
p.add_argument("--output", default="artifacts/monitoring_performance.json")
a = p.parse_args()
df = pd.read_csv(a.input)
required = {"is_churned", "churn_prediction"}
if not required.issubset(df.columns): raise SystemExit(f"Input must contain columns: {sorted(required)}")
y, pred = df.is_churned.astype(int), df.churn_prediction.astype(int)
result = {"rows": len(df), "precision": float(precision_score(y, pred, zero_division=0)),
          "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
          "confusion_matrix": confusion_matrix(y, pred).tolist(),
          "note": "Use only after ground-truth labels are available; feature drift is not a substitute for performance measurement."}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
