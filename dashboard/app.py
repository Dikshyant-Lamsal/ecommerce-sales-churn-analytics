from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from model_comparison import (
    DEFAULT_EXPERIMENT_NAME,
    METRIC_KEYS,
    METRIC_LABELS,
    available_metrics,
    best_model,
    classification_metrics,
    compare_local_to_mlflow,
    comparison_table,
    fetch_mlflow_runs,
    fmt_score,
    local_vs_mlflow_overlap,
    metric_column,
    read_json_checked,
    runs_to_frame,
    safe_uri,
    select_runs,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Sales & Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 E-Commerce Sales & Churn Analytics")
st.caption(
    "Sales intelligence, customer retention, churn modelling, "
    "and MLflow experiment tracking."
)

ARTIFACTS_DIR = Path(
    os.getenv(
        "MODEL_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

# ============================================================
# DATABASE CONNECTION
# ============================================================


@st.cache_resource
def get_engine() -> Engine:
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return create_engine(database_url, pool_pre_ping=True)

    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "ecommerce_dw")
    user = os.getenv("DB_USER", "etl_user")
    password = os.getenv("DB_PASSWORD", "etl_pass")

    url = (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )

    return create_engine(url, pool_pre_ping=True)


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as connection:
        return pd.read_sql(
            text(sql),
            connection,
            params=params or {},
        )


# ============================================================
# DATA ACCESS
# ============================================================


@st.cache_data(ttl=60, show_spinner=False)
def load_sales_data() -> pd.DataFrame:
    """
    Loads sales from the documented star schema.

    fact_sales is at invoice-line grain.
    """
    sql = """
        SELECT
            fs.invoice_no,
            dd.full_date::date AS sale_date,
            dp.description AS product,
            dp.stock_code,
            dg.country,
            dc.customer_id,
            fs.quantity,
            fs.unit_price,
            fs.line_revenue
        FROM fact_sales fs
        LEFT JOIN dim_date dd
            ON fs.date_key = dd.date_key
        LEFT JOIN dim_product dp
            ON fs.product_key = dp.product_key
        LEFT JOIN dim_geography dg
            ON fs.geography_key = dg.geography_key
        LEFT JOIN dim_customer dc
            ON fs.customer_key = dc.customer_key
        WHERE dd.full_date IS NOT NULL
    """
    df = query_df(sql)

    if not df.empty:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        for column in ["quantity", "unit_price", "line_revenue"]:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_customer_data() -> pd.DataFrame:
    """
    Loads customer-level RFM and churn information.
    """
    sql = """
        SELECT
            customer_key,
            customer_id,
            country,
            recency_days,
            frequency,
            monetary,
            customer_segment,
            last_purchase_date
        FROM dim_customer
        WHERE customer_id IS NOT NULL
          AND customer_id <> -1
    """
    df = query_df(sql)

    for column in ["recency_days", "frequency", "monetary"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_retention_data() -> pd.DataFrame:
    sql = """
        SELECT
            dd.full_date::date AS analysis_date,
            fcr.recency_days,
            fcr.frequency,
            fcr.monetary,
            fcr.is_repeat_customer,
            fcr.is_churned,
            dc.country,
            dc.customer_id
        FROM fact_customer_retention fcr
        LEFT JOIN dim_date dd
            ON fcr.analysis_date_key = dd.date_key
        LEFT JOIN dim_customer dc
            ON fcr.customer_key = dc.customer_key
    """
    df = query_df(sql)

    for column in ["recency_days", "frequency", "monetary"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# ============================================================
# MODEL REPORTS & MLFLOW COMPARISON
# (independent of PostgreSQL; rendered even if the database fails)
# ============================================================

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)

VALIDATION_FILES = {
    "Logistic Regression": "logistic_regression_validation_report.json",
    "Random Forest": "random_forest_validation_report.json",
    "XGBoost": "xgboost_validation_report.json",
}
OPTIONAL_REPORT_MODELS = {"XGBoost"}  # only shown if its report exists


@st.cache_data(ttl=60, show_spinner=False)
def load_mlflow_runs(tracking_uri: str, experiment_name: str) -> dict:
    """Cached wrapper; the sidebar and section refresh buttons clear it."""
    return fetch_mlflow_runs(tracking_uri, experiment_name)


def metric_bar_chart(long_df: pd.DataFrame, title: str, show_run_id: bool) -> go.Figure:
    """Grouped bar chart shared by the local-report and MLflow comparisons."""
    fig = px.bar(
        long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        hover_data=["Run ID"] if show_run_id else None,
        title=title,
        range_y=[0, 1.05],
        category_orders={"Metric": [METRIC_LABELS[m] for m in METRIC_KEYS]},
    )
    hover = "<b>%{x}</b><br>%{fullData.name}: %{y:.3f}"
    if show_run_id:
        hover += "<br>Run: %{customdata[0]}"
    fig.update_traces(
        texttemplate="%{y:.3f}",
        textposition="outside",
        hovertemplate=hover + "<extra></extra>",
    )
    fig.update_layout(yaxis_title="Score (0–1)", legend_title_text="Metric")
    return fig


def display_scores(frame: pd.DataFrame, metric_labels: list[str]) -> pd.DataFrame:
    """Copy of ``frame`` with scores as 3-decimal text and '—' for missing."""
    shown = frame.copy()
    for label in metric_labels:
        if label in shown.columns:
            shown[label] = shown[label].map(fmt_score)
    return shown


def load_local_validation() -> tuple[pd.DataFrame, list[str]]:
    records, problems = [], []
    for model_name, filename in VALIDATION_FILES.items():
        report, problem = read_json_checked(ARTIFACTS_DIR / filename)
        if report is None:
            optional_and_absent = (
                model_name in OPTIONAL_REPORT_MODELS
                and not (ARTIFACTS_DIR / filename).exists()
            )
            if not optional_and_absent:
                problems.append(problem)
            continue
        row = {"Model": model_name}
        row.update(classification_metrics(report))
        records.append(row)
    return pd.DataFrame(records), problems


def render_local_reports() -> tuple[pd.DataFrame, dict | None]:
    st.header("Model Evaluation Reports")
    st.caption(
        f"Source: JSON files written by `ml/train.py` in `{ARTIFACTS_DIR}`. "
        "These are separate from the live MLflow data further down."
    )

    # ---- Validation ------------------------------------------------------
    st.subheader("Validation results (used to choose the model)")
    st.caption(
        "Evaluated on the validation snapshots. Precision, recall and F1-score "
        "are for the churn class (label 1); accuracy covers both classes. "
        "ROC-AUC is not part of these reports. XGBoost appears only if its "
        "report file exists."
    )

    validation_df, problems = load_local_validation()
    for problem in problems:
        st.warning(problem)

    if not validation_df.empty:
        local_labels = [
            m for m in ["Accuracy", "Precision", "Recall", "F1-score"]
            if m in validation_df.columns
        ]
        chosen = st.multiselect(
            "Validation metrics",
            local_labels,
            default=local_labels,
            key="local_validation_metrics",
        )
        if chosen:
            long_df = validation_df.melt(
                id_vars=["Model"],
                value_vars=chosen,
                var_name="Metric",
                value_name="Score",
            ).dropna(subset=["Score"])
            if not long_df.empty:
                st.plotly_chart(
                    metric_bar_chart(
                        long_df,
                        "Validation performance by model (local reports)",
                        show_run_id=False,
                    ),
                    use_container_width=True,
                )
        st.dataframe(
            display_scores(validation_df, local_labels),
            use_container_width=True,
            hide_index=True,
        )
    elif not problems:
        st.info(
            f"No validation reports were found in {ARTIFACTS_DIR}. "
            "Check MODEL_ARTIFACTS_DIR and that `python -m ml.train` has been run."
        )

    # ---- Test ------------------------------------------------------------
    evaluation, eval_problem = read_json_checked(ARTIFACTS_DIR / "evaluation_report.json")
    if evaluation:
        st.subheader("Selected model: test evaluation (held-out test snapshots)")
        st.caption(
            "Reported once, for the model chosen on validation F1. These are TEST "
            "metrics on different data from the validation table above, so the two "
            "should not be compared like-for-like, and other models have no test result."
        )

        selected_name = evaluation.get("selected_model", "Unknown")
        test_metrics = evaluation.get("test_metrics", {})
        st.write(f"Selected model: **{selected_name}**")

        rows_info = [
            f"{label}: {evaluation[key]:,}"
            for label, key in [("train rows", "train_rows"),
                               ("validation rows", "validation_rows"),
                               ("test rows", "test_rows")]
            if isinstance(evaluation.get(key), int)
        ]
        if rows_info:
            st.caption(" · ".join(rows_info))

        if test_metrics:
            metric_cols = st.columns(min(len(test_metrics), 5))
            for index, (name, value) in enumerate(test_metrics.items()):
                with metric_cols[index % len(metric_cols)]:
                    st.metric(name.replace("_", " ").upper() if name == "roc_auc"
                              else name.replace("_", " ").title(), fmt_score(value))

        confusion = evaluation.get("confusion_matrix")
        if (
            isinstance(confusion, list)
            and len(confusion) == 2
            and all(isinstance(row, list) and len(row) == 2 for row in confusion)
        ):
            confusion_df = pd.DataFrame(
                confusion,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )
            fig = px.imshow(
                confusion_df,
                text_auto=True,
                aspect="auto",
                title="Selected model confusion matrix (test set)",
                labels={"x": "Predicted label", "y": "Actual label", "color": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)
    elif (ARTIFACTS_DIR / "evaluation_report.json").exists():
        st.warning(eval_problem)
    else:
        st.info("No evaluation_report.json found, so there is no test evaluation to show.")

    return validation_df, evaluation


def render_mlflow_comparison(
    local_validation: pd.DataFrame, local_evaluation: dict | None
) -> None:
    st.header("🧪 MLflow Model Comparison")
    st.caption(
        f"Tracking server: `{safe_uri(MLFLOW_TRACKING_URI)}` · "
        f"Experiment: `{MLFLOW_EXPERIMENT_NAME}`"
    )

    if st.button("🔄 Refresh MLflow data", key="mlflow_refresh"):
        load_mlflow_runs.clear()
        st.rerun()

    result = load_mlflow_runs(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    st.caption(f"MLflow queried at {result['fetched_at']} UTC (cached for 60 seconds).")

    status = result["status"]
    if status != "ok":
        titles = {
            "invalid_uri": "The MLflow tracking URI is not usable.",
            "missing_dependency": "The mlflow package is not available in this container.",
            "unreachable": "The MLflow server could not be reached.",
            "api_error": "The MLflow server returned an error.",
            "no_experiment": "The MLflow experiment does not exist yet.",
        }
        show = st.warning if status in {"unreachable", "no_experiment"} else st.error
        show(titles.get(status, "MLflow data could not be loaded."))
        st.code(result["message"])
        if status == "unreachable" and any(
            host in MLFLOW_TRACKING_URI for host in ("localhost", "127.0.0.1")
        ):
            st.info(
                "Inside the dashboard container, localhost is the dashboard itself. "
                "Set MLFLOW_TRACKING_URI=http://mlflow:5000 on the dashboard service "
                "in docker-compose.yml."
            )
        return

    frame = runs_to_frame(result["runs"])
    if frame.empty:
        st.info(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' exists but has no runs yet.")
        return

    finished = frame["status"] == "FINISHED"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Runs", len(frame))
    k2.metric("Finished", int(finished.sum()))
    k3.metric("Not finished", int((~finished).sum()))
    k4.metric(
        "Training / evaluation runs",
        f"{int((frame['run_type'] == 'training').sum())} / "
        f"{int((frame['run_type'] == 'evaluation').sum())}",
    )

    training = frame[(frame["run_type"] == "training") & finished]
    available = available_metrics(training, "validation")

    # ---- A-C: validation comparison ---------------------------------------
    st.subheader("Model comparison on validation metrics")
    st.caption(
        "Only finished training runs that logged `validation_*` metrics are compared, "
        "so every bar below is on the same split. Test-evaluation runs are shown "
        "separately and never mixed in."
    )

    selected = pd.DataFrame()
    if not available:
        logged = sorted(
            c for c in frame.columns
            if c not in {"run_id", "short_id", "run_name", "model", "model_label",
                         "run_type", "status", "start_time", "params", "tags",
                         "artifact_uri"}
        )
        st.info(
            "No finished training run logged comparable validation metrics. Expected "
            "keys: " + ", ".join(f"`{metric_column('validation', m)}`" for m in METRIC_KEYS)
            + ". Metric keys found in this experiment: "
            + (", ".join(f"`{c}`" for c in logged) if logged else "none")
            + "."
        )
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            criterion = st.selectbox(
                "Comparison criterion",
                available,
                index=available.index("f1") if "f1" in available else 0,
                format_func=lambda m: METRIC_LABELS[m],
                key="mlflow_criterion",
            )
        with c2:
            policy_label = st.radio(
                "Which run represents each model?",
                [
                    "Latest finished run",
                    "Best finished run by the selected criterion",
                ],
                key="mlflow_policy",
                horizontal=True,
            )
        policy = "best" if policy_label.startswith("Best") else "latest"

        selected, notes = select_runs(frame, policy, criterion)
        for note in notes:
            st.caption(f"ℹ️ {note}")

        if selected.empty:
            st.info("No runs satisfy the selected rule.")
        else:
            rule = (
                f"highest validation {METRIC_LABELS[criterion]}"
                if policy == "best" else "most recently started"
            )
            st.caption(
                f"Selection rule: for each model, the finished run with the {rule}."
            )
            multi = selected[selected["n_runs"] > 1]
            if not multi.empty:
                st.caption(
                    "Models with several finished runs: "
                    + ", ".join(f"{r.model_label} ({int(r.n_runs)})" for r in multi.itertuples())
                    + ". `ml/train.py` does not log the snapshot dates, so reruns on "
                      "different data are not distinguishable here."
                )

            table = comparison_table(selected, "validation", available)
            metric_labels = [METRIC_LABELS[m] for m in available]
            st.dataframe(
                display_scores(table, metric_labels),
                use_container_width=True,
                hide_index=True,
            )

            chosen = st.multiselect(
                "Metrics to chart",
                metric_labels,
                default=metric_labels,
                key="mlflow_chart_metrics",
            )
            if chosen:
                long_df = table.melt(
                    id_vars=["Model", "Run ID"],
                    value_vars=chosen,
                    var_name="Metric",
                    value_name="Score",
                )
                absent = long_df[long_df["Score"].isna()]
                long_df = long_df.dropna(subset=["Score"])
                if not long_df.empty:
                    st.plotly_chart(
                        metric_bar_chart(
                            long_df,
                            f"MLflow validation metrics by model ({rule})",
                            show_run_id=True,
                        ),
                        use_container_width=True,
                    )
                if not absent.empty:
                    st.caption(
                        "Not logged (left blank, not plotted as zero): "
                        + ", ".join(f"{r.Model} – {r.Metric}" for r in absent.itertuples())
                    )

            st.markdown("**Best-model summary**")
            verdict = best_model(selected, "validation", criterion)
            if verdict["valid"] and verdict["winner"]:
                st.success(f"Criterion: validation {METRIC_LABELS[criterion]}. {verdict['message']}")
            elif verdict["valid"]:
                st.info(f"Criterion: validation {METRIC_LABELS[criterion]}. {verdict['message']}")
            else:
                st.warning(
                    f"No winner for validation {METRIC_LABELS[criterion]}: {verdict['message']}"
                )
            st.caption(
                "Leading on one metric does not make a model best for every business "
                "goal. Recall favours catching churners, precision favours avoiding "
                "wasted retention offers, and small gaps on a short dataset may not "
                "be meaningful."
            )

            diffs = compare_local_to_mlflow(local_validation, selected)
            overlap = local_vs_mlflow_overlap(local_validation, selected)
            if diffs:
                st.info(
                    "The local validation reports and the selected MLflow runs differ "
                    "(" + "; ".join(diffs) + "). They may come from different training "
                    "runs, since each `python -m ml.train` run rewrites the local JSON "
                    "and adds new MLflow runs. Neither source has been overwritten."
                )
            elif overlap:
                st.caption(
                    f"The local validation reports agree with the selected MLflow runs "
                    f"on {overlap} metric values."
                )

    # ---- Test evaluation runs ---------------------------------------------
    st.subheader("Test evaluation runs (selected model only)")
    evaluation_runs = frame[(frame["run_type"] == "evaluation") & finished].sort_values(
        "start_time", ascending=False, na_position="last"
    )
    if evaluation_runs.empty:
        st.info("No finished test-evaluation run found.")
    else:
        st.caption(
            "Each run reports TEST metrics for one model chosen on validation F1. "
            "Different split from the validation comparison; not a model-vs-model comparison."
        )
        test_available = available_metrics(evaluation_runs, "test")
        test_rows = []
        for _, run in evaluation_runs.iterrows():
            row = {
                "Run ID": run["short_id"],
                "Started": run["start_time"],
                "Selected model": run["params"].get("selected_model", "—"),
                "Selected run": str(run["params"].get("selected_run_id", "—"))[:8],
            }
            for metric in test_available:
                row[METRIC_LABELS[metric]] = run[metric_column("test", metric)]
            test_rows.append(row)
        st.dataframe(
            display_scores(pd.DataFrame(test_rows), [METRIC_LABELS[m] for m in test_available]),
            use_container_width=True,
            hide_index=True,
        )
        if local_evaluation and isinstance(local_evaluation.get("test_metrics"), dict):
            latest = evaluation_runs.iloc[0]
            differing = [
                METRIC_LABELS[m]
                for m in METRIC_KEYS
                if isinstance(local_evaluation["test_metrics"].get(m), (int, float))
                and metric_column("test", m) in evaluation_runs.columns
                and pd.notna(latest[metric_column("test", m)])
                and abs(local_evaluation["test_metrics"][m] - latest[metric_column("test", m)]) > 1e-6
            ]
            if differing:
                st.info(
                    "evaluation_report.json differs from the newest MLflow test run for "
                    + ", ".join(differing)
                    + ". They may come from different training runs."
                )

    # ---- D: run details ---------------------------------------------------
    st.subheader("Run details")
    details = frame[["run_name", "run_type", "status", "run_id", "start_time"]].rename(
        columns={"run_name": "Run name", "run_type": "Type", "status": "Status",
                 "run_id": "Run ID", "start_time": "Started"}
    )
    metric_cols_present = []
    for split in ("validation", "test"):
        for metric in METRIC_KEYS:
            col = metric_column(split, metric)
            if col in frame.columns and frame[col].notna().any():
                details[col] = frame[col]
                metric_cols_present.append(col)
    for col in metric_cols_present:
        details[col] = details[col].map(fmt_score)
    st.dataframe(details, use_container_width=True, hide_index=True)

    run_ids = frame["run_id"].tolist()
    selected_run_id = st.selectbox(
        "Inspect a run",
        run_ids,
        format_func=lambda rid: (
            f"{frame.loc[frame['run_id'] == rid, 'run_name'].iloc[0] or 'unnamed'} — {rid[:8]}"
        ),
        key="mlflow_inspect_run",
    )
    run_row = frame[frame["run_id"] == selected_run_id].iloc[0]
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Parameters**")
        st.json(run_row["params"] or {"info": "no parameters logged"}, expanded=False)
    with d2:
        st.markdown("**Logged metrics**")
        logged_metrics = {
            c: float(run_row[c])
            for c in frame.columns
            if c.startswith(("validation_", "test_")) and pd.notna(run_row[c])
        }
        st.json(logged_metrics or {"info": "no metrics logged"}, expanded=False)

    if st.button("List artifacts for this run", key="mlflow_list_artifacts"):
        try:
            from mlflow.exceptions import MlflowException
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            found, pending = [], [None]
            while pending:
                for artifact in client.list_artifacts(selected_run_id, pending.pop()):
                    found.append({
                        "Path": artifact.path,
                        "Type": "Directory" if artifact.is_dir else "File",
                        "Size (bytes)": getattr(artifact, "file_size", None),
                    })
                    if artifact.is_dir:
                        pending.append(artifact.path)
            if found:
                st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
            else:
                st.info("This run has no logged artifacts.")
        except ImportError as exc:
            st.error(f"mlflow could not be imported: {exc}")
        except MlflowException as exc:
            st.error("Could not list artifacts for this run.")
            st.code(safe_uri(str(exc))[:500])


def render_model_sections() -> None:
    st.divider()
    local_validation, local_evaluation = render_local_reports()
    st.divider()
    render_mlflow_comparison(local_validation, local_evaluation)


# ============================================================
# SIDEBAR / GLOBAL FILTERS
# ============================================================

with st.sidebar:
    st.header("Dashboard Controls")

    if st.button("🔄 Refresh dashboard", use_container_width=True):
        load_sales_data.clear()
        load_customer_data.clear()
        load_retention_data.clear()
        load_mlflow_runs.clear()
        st.rerun()

    st.caption("Data is cached for 60 seconds.")

# ============================================================
# LOAD DATA
# ============================================================

try:
    with st.spinner("Loading data from PostgreSQL..."):
        sales = load_sales_data()
        customers = load_customer_data()
        retention = load_retention_data()

except Exception as exc:
    st.error("Could not load analytics data from PostgreSQL.")
    st.info(
        "Check that PostgreSQL is running, the database credentials are "
        "correct, and the documented tables exist."
    )
    st.code(str(exc))
    st.warning("Sales and customer sections are unavailable; model sections below still work.")
    render_model_sections()
    st.stop()

if sales.empty:
    st.warning("No sales rows were returned from fact_sales.")
    render_model_sections()
    st.stop()

# ============================================================
# FILTERS
# ============================================================

sales = sales.dropna(subset=["sale_date"]).copy()

with st.sidebar:
    st.subheader("Sales Filters")

    min_date = sales["sale_date"].min().date()
    max_date = sales["sale_date"].max().date()

    date_range = st.date_input(
        "Sales date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    countries = sorted(sales["country"].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Countries",
        countries,
        default=[],
        help="Leave empty to include all countries.",
    )

    if "product" in sales.columns:
        product_values = sorted(
            sales["product"].dropna().astype(str).unique().tolist()
        )
        selected_products = st.multiselect(
            "Products",
            product_values,
            default=[],
        )
    else:
        selected_products = []

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    sales = sales[
        sales["sale_date"].dt.date.between(start_date, end_date)
    ]

if selected_countries:
    sales = sales[sales["country"].isin(selected_countries)]

if selected_products and "product" in sales.columns:
    sales = sales[sales["product"].astype(str).isin(selected_products)]

# ============================================================
# OVERVIEW KPIs
# ============================================================

st.header("Sales Overview")

total_revenue = sales["line_revenue"].sum()
total_orders = sales["invoice_no"].nunique()
unique_customers = sales.loc[
    sales["customer_id"].notna() & (sales["customer_id"] != -1),
    "customer_id",
].nunique()

total_units = sales["quantity"].sum()
average_order_value = (
    total_revenue / total_orders if total_orders else 0
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Revenue", f"£{total_revenue:,.2f}")
k2.metric("Orders", f"{total_orders:,}")
k3.metric("Unique Customers", f"{unique_customers:,}")
k4.metric("Units Sold", f"{total_units:,.0f}")
k5.metric("Average Order Value", f"£{average_order_value:,.2f}")

# ============================================================
# SALES TRENDS
# ============================================================

st.divider()
st.header("Revenue & Order Trends")

daily = (
    sales.groupby(sales["sale_date"].dt.date)
    .agg(
        revenue=("line_revenue", "sum"),
        orders=("invoice_no", "nunique"),
    )
    .reset_index()
    .rename(columns={"sale_date": "date"})
)

left, right = st.columns(2)

with left:
    fig = px.line(
        daily,
        x="date",
        y="revenue",
        markers=True,
        title="Daily Revenue",
        labels={"date": "Date", "revenue": "Revenue (£)"},
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(
        daily,
        x="date",
        y="orders",
        markers=True,
        title="Daily Orders",
        labels={"date": "Date", "orders": "Orders"},
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

monthly = sales.copy()
monthly["month"] = monthly["sale_date"].dt.to_period("M").astype(str)

monthly = (
    monthly.groupby("month")
    .agg(
        revenue=("line_revenue", "sum"),
        orders=("invoice_no", "nunique"),
    )
    .reset_index()
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=monthly["month"],
        y=monthly["revenue"],
        name="Revenue (£)",
    )
)
fig.update_layout(
    title="Monthly Revenue",
    xaxis_title="Month",
    yaxis_title="Revenue (£)",
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PRODUCT ANALYTICS
# ============================================================

st.divider()
st.header("Product Performance")

if "product" in sales.columns:
    product_sales = (
        sales.dropna(subset=["product"])
        .groupby("product", as_index=False)
        .agg(
            revenue=("line_revenue", "sum"),
            units=("quantity", "sum"),
            orders=("invoice_no", "nunique"),
        )
        .sort_values("revenue", ascending=False)
    )

    top_n = st.slider(
        "Number of products to display",
        min_value=5,
        max_value=30,
        value=10,
    )

    top_products = product_sales.head(top_n)

    fig = px.bar(
        top_products.sort_values("revenue"),
        x="revenue",
        y="product",
        orientation="h",
        title=f"Top {top_n} Products by Revenue",
        labels={"revenue": "Revenue (£)", "product": "Product"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View product sales table"):
        st.dataframe(
            product_sales,
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Product descriptions are unavailable in the sales query.")

# ============================================================
# COUNTRY ANALYTICS
# ============================================================

st.divider()
st.header("Geographic Sales")

country_sales = (
    sales.dropna(subset=["country"])
    .groupby("country", as_index=False)
    .agg(
        revenue=("line_revenue", "sum"),
        orders=("invoice_no", "nunique"),
    )
    .sort_values("revenue", ascending=False)
)

if not country_sales.empty:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            country_sales.head(15).sort_values("revenue"),
            x="revenue",
            y="country",
            orientation="h",
            title="Top Countries by Revenue",
            labels={"revenue": "Revenue (£)", "country": "Country"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(
            country_sales.head(10),
            names="country",
            values="revenue",
            title="Revenue Share: Top 10 Countries",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View country-level sales"):
        st.dataframe(
            country_sales,
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

st.divider()
st.header("Customer Segmentation (RFM)")

if not customers.empty and "customer_segment" in customers.columns:
    segment_data = customers.dropna(subset=["customer_segment"]).copy()

    segment_counts = (
        segment_data.groupby("customer_segment")
        .size()
        .reset_index(name="customers")
        .sort_values("customers", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            segment_counts,
            x="customer_segment",
            y="customers",
            title="Customers by RFM Segment",
            labels={
                "customer_segment": "Segment",
                "customers": "Customers",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(
            segment_counts,
            names="customer_segment",
            values="customers",
            title="Customer Segment Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    if {"frequency", "monetary"}.issubset(segment_data.columns):
        fig = px.scatter(
            segment_data,
            x="frequency",
            y="monetary",
            color="customer_segment",
            hover_data=["customer_id", "recency_days"],
            title="Customer Frequency vs Monetary Value",
            labels={
                "frequency": "Purchase Frequency",
                "monetary": "Monetary Value (£)",
                "recency_days": "Recency (days)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View customer RFM data"):
        st.dataframe(
            segment_data,
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Customer segmentation data is unavailable.")

# ============================================================
# CUSTOMER RETENTION & CHURN
# ============================================================

st.divider()
st.header("Customer Retention & Churn")

if not retention.empty:
    r1, r2, r3 = st.columns(3)

    total_retention_customers = len(retention)

    if "is_churned" in retention.columns:
        churned = retention["is_churned"].fillna(False).astype(bool)
        churn_count = int(churned.sum())
        churn_rate = (
            churn_count / total_retention_customers * 100
            if total_retention_customers
            else 0
        )
        r1.metric("Churned Customers", f"{churn_count:,}")
        r2.metric("Churn Rate", f"{churn_rate:.2f}%")
    else:
        churn_count = None
        r1.metric("Churned Customers", "N/A")
        r2.metric("Churn Rate", "N/A")

    if "is_repeat_customer" in retention.columns:
        repeat = (
            retention["is_repeat_customer"]
            .fillna(False)
            .astype(bool)
        )
        repeat_count = int(repeat.sum())
        repeat_rate = (
            repeat_count / total_retention_customers * 100
            if total_retention_customers
            else 0
        )
        r3.metric("Repeat Customer Rate", f"{repeat_rate:.2f}%")
    else:
        r3.metric("Repeat Customer Rate", "N/A")

    c1, c2 = st.columns(2)

    if "is_churned" in retention.columns:
        churn_chart = retention.copy()
        churn_chart["Status"] = (
            churn_chart["is_churned"]
            .fillna(False)
            .astype(bool)
            .map({True: "Churned", False: "Active"})
        )

        churn_summary = (
            churn_chart.groupby("Status")
            .size()
            .reset_index(name="Customers")
        )

        with c1:
            fig = px.pie(
                churn_summary,
                names="Status",
                values="Customers",
                title="Active vs Churned Customers",
                hole=0.45,
            )
            st.plotly_chart(fig, use_container_width=True)

    if "is_repeat_customer" in retention.columns:
        repeat_chart = retention.copy()
        repeat_chart["Purchase Type"] = (
            repeat_chart["is_repeat_customer"]
            .fillna(False)
            .astype(bool)
            .map({True: "Repeat", False: "One-time"})
        )

        repeat_summary = (
            repeat_chart.groupby("Purchase Type")
            .size()
            .reset_index(name="Customers")
        )

        with c2:
            fig = px.bar(
                repeat_summary,
                x="Purchase Type",
                y="Customers",
                title="Repeat vs One-Time Customers",
            )
            st.plotly_chart(fig, use_container_width=True)

    if {"recency_days", "monetary"}.issubset(retention.columns):
        fig = px.scatter(
            retention,
            x="recency_days",
            y="monetary",
            color=(
                "is_churned"
                if "is_churned" in retention.columns
                else None
            ),
            hover_data=["frequency"],
            title="Recency vs Customer Monetary Value",
            labels={
                "recency_days": "Recency (days)",
                "monetary": "Monetary Value (£)",
                "is_churned": "Churned",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Customer retention data is unavailable.")

# ============================================================
# MODEL REPORTS & MLFLOW COMPARISON
# ============================================================

render_model_sections()

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "E-Commerce Sales & Churn Analytics · PostgreSQL · "
    "Streamlit · Plotly · MLflow"
)