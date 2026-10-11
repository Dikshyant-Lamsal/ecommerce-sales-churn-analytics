"""Compare incoming customer feature distributions with training reference data."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ml.features import FEATURE_COLUMNS

REFERENCE = Path(os.getenv("DRIFT_REFERENCE_PATH", "artifacts/feature_reference.csv"))
CURRENT = Path(os.getenv("DRIFT_CURRENT_PATH", "data/monitoring/current_features.csv"))
OUTPUT = Path(os.getenv("DRIFT_REPORT_PATH", "artifacts/drift_report.json"))


def compare(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = {"reference_rows": len(reference), "current_rows": len(current), "features": {}, "drift_detected": False,
              "method": "two-sample Kolmogorov-Smirnov test; alpha=0.05 (screening signal, not proof of model degradation)"}
    for col in FEATURE_COLUMNS:
        if col not in reference or col not in current:
            report["features"][col] = {"status": "missing_column"}
            report["drift_detected"] = True
            continue
        a = pd.to_numeric(reference[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            report["features"][col] = {"status": "insufficient_data", "reference_non_null": len(a), "current_non_null": len(b)}
            continue
        test = ks_2samp(a, b)
        drift = bool(test.pvalue < 0.05)
        report["features"][col] = {"ks_statistic": float(test.statistic), "p_value": float(test.pvalue),
                                   "reference_missing_rate": float(1 - len(a) / len(reference)),
                                   "current_missing_rate": float(1 - len(b) / len(current)), "drift_detected": drift}
        report["drift_detected"] |= drift
    return report


def main():
    if not REFERENCE.exists(): raise FileNotFoundError(f"Reference not found: {REFERENCE}; run training first")
    if not CURRENT.exists(): raise FileNotFoundError(f"Current feature file not found: {CURRENT}; provide recent customer features")
    report = compare(pd.read_csv(REFERENCE), pd.read_csv(CURRENT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
