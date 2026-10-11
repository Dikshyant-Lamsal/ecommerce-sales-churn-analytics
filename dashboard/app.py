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
# SIDEBAR / GLOBAL FILTERS
# ============================================================

with st.sidebar:
    st.header("Dashboard Controls")

    if st.button("🔄 Refresh dashboard", use_container_width=True):
        load_sales_data.clear()
        load_customer_data.clear()
        load_retention_data.clear()
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
    st.stop()

if sales.empty:
    st.warning("No sales rows were returned from fact_sales.")
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
# MODEL REPORTS FROM LOCAL ARTIFACTS
# ============================================================

st.divider()
st.header("Model Evaluation Reports")
st.caption(
    "These reports are read from the artifacts directory. "
    "They are separate from the live MLflow metrics below."
)


def read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            return result if isinstance(result, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def classification_metrics(report: dict) -> dict:
    positive = report.get("1", {})
    result = {}

    if "accuracy" in report:
        result["Accuracy"] = report["accuracy"]

    for output_name, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1-score"),
    ]:
        if key in positive:
            result[output_name] = positive[key]

    return result


validation_files = {
    "Logistic Regression": "logistic_regression_validation_report.json",
    "Random Forest": "random_forest_validation_report.json",
    "XGBoost": "xgboost_validation_report.json",
}

validation_records = []

for model_name, filename in validation_files.items():
    report = read_json(ARTIFACTS_DIR / filename)
    if report:
        row = {"Model": model_name}
        row.update(classification_metrics(report))
        validation_records.append(row)

if validation_records:
    validation_df = pd.DataFrame(validation_records)

    metric_options = [
        metric
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]
        if metric in validation_df.columns
    ]

    chosen_metrics = st.multiselect(
        "Validation metrics",
        metric_options,
        default=metric_options,
    )

    if chosen_metrics:
        chart_data = validation_df.melt(
            id_vars=["Model"],
            value_vars=chosen_metrics,
            var_name="Metric",
            value_name="Score",
        ).dropna(subset=["Score"])

        fig = px.bar(
            chart_data,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Validation Performance by Model",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        f"No validation reports were found in {ARTIFACTS_DIR}. "
        "Check MODEL_ARTIFACTS_DIR and the report filenames."
    )

evaluation = read_json(ARTIFACTS_DIR / "evaluation_report.json")

if evaluation:
    st.subheader("Selected Model — Test Evaluation")

    selected_name = evaluation.get("selected_model", "Unknown")
    test_metrics = evaluation.get("test_metrics", {})

    st.write(f"Selected model: **{selected_name}**")

    if test_metrics:
        metric_cols = st.columns(min(len(test_metrics), 5))

        for index, (name, value) in enumerate(test_metrics.items()):
            with metric_cols[index % len(metric_cols)]:
                try:
                    st.metric(name.replace("_", " ").title(), f"{float(value):.3f}")
                except (ValueError, TypeError):
                    st.metric(name.replace("_", " ").title(), str(value))

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
            title="Selected Model Confusion Matrix",
            labels={
                "x": "Predicted label",
                "y": "Actual label",
                "color": "Count",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# LIVE MLFLOW EXPERIMENT TRACKING
# ============================================================

st.divider()
st.header("🧪 MLflow Experiment Tracking")
st.caption(
    f"Tracking server: {MLFLOW_TRACKING_URI} · "
    "Experiment: ecommerce-customer-churn"
)


@st.cache_data(ttl=30, show_spinner=False)
def load_mlflow_data(tracking_uri: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = client.search_experiments()
    experiment = next(
        (
            item for item in experiments
            if item.name == "ecommerce-customer-churn"
        ),
        None,
    )

    if experiment is None:
        return [], "Experiment 'ecommerce-customer-churn' was not found."

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=200,
        order_by=["attributes.start_time DESC"],
    )

    records = []

    for run in runs:
        record = {
            "Run Name": run.data.tags.get("mlflow.runName", "Unnamed run"),
            "Run ID": run.info.run_id,
            "Status": run.info.status,
            "Start Time": (
                pd.to_datetime(run.info.start_time, unit="ms", utc=True)
                if run.info.start_time
                else pd.NaT
            ),
            "Artifact URI": run.info.artifact_uri,
        }

        record.update(
            {f"param:{key}": value for key, value in run.data.params.items()}
        )
        record.update(run.data.metrics)
        records.append(record)

    return records, None


try:
    mlflow_records, mlflow_error = load_mlflow_data(MLFLOW_TRACKING_URI)

    if mlflow_error:
        st.warning(mlflow_error)
    elif not mlflow_records:
        st.info("No MLflow runs were found in this experiment.")
    else:
        mlflow_df = pd.DataFrame(mlflow_records)

        successful_runs = mlflow_df[
            mlflow_df["Status"].astype(str).str.upper() == "FINISHED"
        ]

        a, b, c = st.columns(3)
        a.metric("Total Runs", len(mlflow_df))
        b.metric("Finished Runs", len(successful_runs))
        c.metric(
            "Failed Runs",
            int(
                mlflow_df["Status"]
                .astype(str)
                .str.upper()
                .eq("FAILED")
                .sum()
            ),
        )

        # Model names are derived from the actual MLflow run names.
        st.subheader("Run History")

        history_df = mlflow_df.copy()
        history_df["Run Label"] = (
            history_df["Run Name"]
            + " · "
            + history_df["Run ID"].str[:8]
        )

        metric_columns = [
            column
            for column in history_df.columns
            if not column.startswith("param:")
            and column not in {
                "Run Name",
                "Run ID",
                "Status",
                "Start Time",
                "Artifact URI",
                "Run Label",
            }
            and pd.api.types.is_numeric_dtype(history_df[column])
            and history_df[column].notna().any()
        ]

        if metric_columns:
            selected_history_metric = st.selectbox(
                "Metric to plot across runs",
                metric_columns,
                key="mlflow_history_metric",
            )

            plot_df = history_df.dropna(
                subset=[selected_history_metric]
            )

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x="Run Label",
                    y=selected_history_metric,
                    color="Run Name",
                    hover_data=["Run ID", "Status"],
                    title=f"MLflow Run History: {selected_history_metric}",
                )
                fig.update_layout(xaxis_title="Run")
                st.plotly_chart(fig, use_container_width=True)

            # Latest finished run per model name for fairer comparison.
            comparison_df = successful_runs.copy()
            comparison_df = comparison_df.sort_values(
                "Start Time",
                ascending=False,
                na_position="last",
            )
            comparison_df = comparison_df.drop_duplicates(
                subset=["Run Name"],
                keep="first",
            )

            available_comparison_metrics = [
                column
                for column in metric_columns
                if comparison_df[column].notna().any()
            ]

            st.subheader("Latest Finished Run per Model")

            if available_comparison_metrics:
                selected_comparison_metrics = st.multiselect(
                    "Metrics to compare",
                    available_comparison_metrics,
                    default=available_comparison_metrics[:4],
                    key="mlflow_comparison_metrics",
                )

                if selected_comparison_metrics:
                    comparison_long = comparison_df.melt(
                        id_vars=["Run Name", "Run ID"],
                        value_vars=selected_comparison_metrics,
                        var_name="Metric",
                        value_name="Score",
                    ).dropna(subset=["Score"])

                    fig = px.bar(
                        comparison_long,
                        x="Run Name",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        hover_data=["Run ID"],
                        title="Latest Model Run Comparison",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "No comparable numeric metrics are logged for "
                    "finished runs. Inspect the run details below."
                )
        else:
            st.info(
                "No numeric MLflow metrics were found. "
                "Your training script may need to log metrics with "
                "mlflow.log_metric()."
            )

        st.subheader("Run Details")

        display_columns = [
            column
            for column in [
                "Run Name",
                "Run ID",
                "Status",
                "Start Time",
            ]
            if column in mlflow_df.columns
        ]

        st.dataframe(
            mlflow_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View logged parameters and metrics"):
            st.dataframe(
                mlflow_df,
                use_container_width=True,
                hide_index=True,
            )

        selected_run_id = st.selectbox(
            "Inspect run artifacts",
            options=mlflow_df["Run ID"].tolist(),
            format_func=lambda run_id: (
                f"{mlflow_df.loc[mlflow_df['Run ID'] == run_id, 'Run Name'].iloc[0]}"
                f" — {run_id[:8]}"
            ),
        )

        if st.button("List artifacts for selected run"):
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

                def list_artifacts_recursively(run_id: str):
                    found = []
                    pending = [""]

                    while pending:
                        current_path = pending.pop()
                        for artifact in client.list_artifacts(
                            run_id,
                            current_path or None,
                        ):
                            found.append({
                                "Path": artifact.path,
                                "Type": (
                                    "Directory" if artifact.is_dir else "File"
                                ),
                                "Size (bytes)": getattr(
                                    artifact, "file_size", None
                                ),
                            })
                            if artifact.is_dir:
                                pending.append(artifact.path)

                    return found

                artifact_records = list_artifacts_recursively(selected_run_id)

                if artifact_records:
                    st.dataframe(
                        pd.DataFrame(artifact_records),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("This run has no logged artifacts.")

            except Exception as exc:
                st.error("Could not retrieve artifacts for this run.")
                st.code(str(exc))

        if st.button("Refresh MLflow charts"):
            load_mlflow_data.clear()
            st.rerun()

except Exception as exc:
    st.warning(
        "MLflow data could not be loaded. Confirm the tracking URI is "
        "reachable from the dashboard container and MLflow is installed."
    )
    st.code(str(exc))

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "E-Commerce Sales & Churn Analytics · PostgreSQL · "
    "Streamlit · Plotly · MLflow"
)
