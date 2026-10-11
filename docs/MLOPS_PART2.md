# Part 2 — MLOps design and operating guide

## Churn target and leakage controls

- **Prediction unit:** one identified customer at a monthly snapshot date.
- **Target:** `is_churned = 1` when the customer makes no qualifying purchase in the 90 days strictly after the snapshot; `0` when at least one qualifying purchase occurs in that window.
- **Feature window:** 180 days strictly before and including the snapshot.
- **Features:** recency, qualifying order count, total monetary value, average order value, item quantity, customer tenure, and unique products.
- Customers without a complete forward 90-day label window are not used. Guest customers without a CustomerID are excluded from supervised customer-level modeling.
- The source dataset is the UCI Online Retail data described in the Part 1 README. The dataset spans roughly one year, so results are an academic demonstration and may be unstable across time.
- Splits are chronological by snapshot date. Model choice uses validation F1; test metrics are reported after selection. If there are fewer than three distinct snapshot dates, training stops rather than silently switching to a random split.

## Train and track

1. Place the source workbook at `data/landing/Online Retail.xlsx`, or use `data/staging/cleaned_transactions.csv` produced by Part 1.
2. Start MLflow with `docker compose up -d mlflow`.
3. From the project root and active virtual environment, run:

   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   python -m ml.train
   ```

4. Open `http://localhost:5000` to compare Logistic Regression and Random Forest runs. The chosen validation-F1 model is registered as `EcommerceChurnClassifier` and assigned the `champion` alias where registry support is available.
5. Review `artifacts/evaluation_report.json`. Do not select a model based on test-set performance.

## API

```bash
uvicorn api.main:app --reload --port 8000
```

- `GET /health` — service/model status
- `GET /model-info` — features, model reference, and windows
- `GET /metrics` — request count, server errors, and latency summary
- `POST /predict` — validate a customer feature vector and return a class plus estimated churn probability

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"12345","recency_days":45,"orders_180d":3,"monetary_180d":250,"avg_order_value_180d":83.33,"items_180d":12,"tenure_days":240,"unique_products_180d":5}'
```

The probability is the classifier's estimated probability, not a guarantee of future behavior. The Streamlit prediction panel accepts features manually; it does not silently infer features for an arbitrary customer ID.

## Docker deployment

1. Train/register the model first (instructions above).
2. Ensure PostgreSQL from Part 1 is reachable from Docker and `.env` contains the correct database values.
3. Run `docker compose up --build`. Dashboard: `http://localhost:8501`, API: `http://localhost:8000/docs`, MLflow: `http://localhost:5000`.
4. The dashboard container uses `host.docker.internal` for PostgreSQL by default. Override `DB_HOST` if your environment differs.
5. MLflow metadata and artifacts are persisted in `mlflow_data/`; keep this directory backed up. Do not commit credentials or private data.

## Monitoring

- Input validation is enforced by Pydantic; API request count, server errors, mean latency, and p95 latency are exposed at `GET /metrics`.
- Feature drift compares a training reference feature CSV with a current feature CSV using two-sample Kolmogorov–Smirnov tests (alpha 0.05). This is a screening signal, not proof that performance has degraded.
- To run drift monitoring, put recent feature rows (same feature columns as training) at `data/monitoring/current_features.csv`, then run `python -m monitoring.drift`.
- Once verified labels are available, evaluate model precision/recall using a CSV with `is_churned` and `churn_prediction` columns: `python monitoring/performance.py --input path/to/labeled_predictions.csv`.
- Drift monitoring is possible without labels; precision/recall monitoring is not.

## Proposed retraining policy (adjust and justify in the report)

1. Consider retraining monthly after a new validated transaction load, if there are enough newly labeled snapshots.
2. Trigger an investigation if two or more core features show statistically significant drift, or if a key feature's missing rate rises by more than 5 percentage points.
3. When ground-truth labels become available, investigate and retrain if recall or precision falls below a documented business threshold (initial proposal: 0.60 for either metric, subject to validation and class balance).
4. Retrain into a candidate version, compare it on a new chronological validation period, and promote only if it meets the agreed metric thresholds and does not materially regress on the key metric.
5. Keep the previous registered version and record training-data dates, metrics, feature definitions, and the deployment decision.

These thresholds are starter policy proposals, not universal statistical guarantees. Explain their business rationale and revise them based on validation results.

## Known limitations to disclose

- The UCI dataset covers a short historical period; churn labels are generated from observed transaction inactivity and are not supplied ground-truth labels from the retailer.
- Snapshot observations for the same customer can occur in more than one split. The split is chronological to reduce temporal leakage, but customer overlap means the test is not a fully customer-disjoint generalization test.
- The dashboard prediction panel uses manually entered feature values. A production enhancement would calculate these features from the customer mart and send them to the API automatically.
- Local API metrics are in-memory and reset on restart; production monitoring should persist metrics and logs.
- The registry/API should be smoke-tested after training in the target environment; library/server versions and storage configuration can affect MLflow alias support.
