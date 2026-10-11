
import os
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="E-Commerce Sales & Retention Dashboard",
    layout="wide",
)


@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


@st.cache_data(ttl=600)
def run_query(query, params=None):
    engine = get_engine()
    return pd.read_sql(query, engine, params=params)


st.title("📊 E-Commerce Sales Analytics & Customer Retention Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

date_bounds = run_query("""
    SELECT
        MIN(full_date) AS min_d,
        MAX(full_date) AS max_d
    FROM dim_date
    WHERE full_date <= DATE '2011-12-09';
""")

min_date = date_bounds.loc[0, "min_d"]
max_date = date_bounds.loc[0, "max_d"]

# Validate dates before passing them to Streamlit.
if pd.isna(min_date) or pd.isna(max_date):
    st.error(
        "No transaction date range is available in dim_date. "
        "Please verify that the date dimension has been loaded."
    )
    st.stop()

min_date = pd.Timestamp(min_date).date()
max_date = pd.Timestamp(max_date).date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

countries_df = run_query("""
    SELECT DISTINCT country
    FROM dim_geography
    ORDER BY country;
""")

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries_df["country"].tolist(),
    default=[],
)

# Build the country filter safely using query parameters.
country_filter_sql = ""
query_params = {
    "start_date": start_date,
    "end_date": end_date,
}

if selected_countries:
    placeholders = ", ".join(
        [f":country_{i}" for i in range(len(selected_countries))]
    )
    country_filter_sql = f"AND g.country IN ({placeholders})"

    for i, country in enumerate(selected_countries):
        query_params[f"country_{i}"] = country

# --- Top-level KPIs ---
kpi_df = run_query(
    f"""
    SELECT
        COUNT(DISTINCT f.invoice_no) AS total_orders,
        COALESCE(SUM(f.line_revenue), 0) AS total_revenue,
        COUNT(DISTINCT f.customer_key) AS total_customers
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_key = d.date_key
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    JOIN dim_geography g
        ON c.geography_key = g.geography_key
    WHERE d.full_date BETWEEN :start_date AND :end_date
    {country_filter_sql};
    """,
    params=query_params,
)

total_orders = int(kpi_df.loc[0, "total_orders"] or 0)
total_revenue = float(kpi_df.loc[0, "total_revenue"] or 0)
total_customers = int(kpi_df.loc[0, "total_customers"] or 0)

col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Revenue", f"£{total_revenue:,.2f}")
col3.metric("Total Customers", f"{total_customers:,}")

st.divider()

# --- Sales trend ---
st.subheader("📈 Sales Trend")

sales_trend = run_query(
    f"""
    SELECT
        d.full_date,
        COUNT(DISTINCT f.invoice_no) AS total_orders,
        COALESCE(SUM(f.line_revenue), 0) AS revenue
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_key = d.date_key
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    JOIN dim_geography g
        ON c.geography_key = g.geography_key
    WHERE d.full_date BETWEEN :start_date AND :end_date
    {country_filter_sql}
    GROUP BY d.full_date
    ORDER BY d.full_date;
    """,
    params=query_params,
)

if not sales_trend.empty:
    fig_sales = px.line(
        sales_trend,
        x="full_date",
        y="revenue",
        title="Daily Revenue",
        labels={
            "full_date": "Date",
            "revenue": "Revenue (£)",
        },
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    fig_orders = px.line(
        sales_trend,
        x="full_date",
        y="total_orders",
        title="Daily Orders",
        labels={
            "full_date": "Date",
            "total_orders": "Orders",
        },
    )
    st.plotly_chart(fig_orders, use_container_width=True)
else:
    st.info("No sales data is available for the selected filters.")

st.divider()

# --- Revenue by country ---
st.subheader("🌍 Revenue by Country")

country_sales = run_query(
    f"""
    SELECT
        g.country,
        COALESCE(SUM(f.line_revenue), 0) AS revenue,
        COUNT(DISTINCT f.invoice_no) AS total_orders
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_key = d.date_key
    JOIN dim_customer c
        ON f.customer_key = c.customer_key
    JOIN dim_geography g
        ON c.geography_key = g.geography_key
    WHERE d.full_date BETWEEN :start_date AND :end_date
    {country_filter_sql}
    GROUP BY g.country
    ORDER BY revenue DESC;
    """,
    params=query_params,
)

if not country_sales.empty:
    fig_country = px.bar(
        country_sales.head(20),
        x="country",
        y="revenue",
        title="Top 20 Countries by Revenue",
        labels={
            "country": "Country",
            "revenue": "Revenue (£)",
        },
    )
    st.plotly_chart(fig_country, use_container_width=True)
    st.dataframe(country_sales, use_container_width=True)
else:
    st.info("No country sales data is available for the selected filters.")

st.divider()

# --- Customer RFM analysis ---
st.subheader("👥 Customer RFM Analysis")

rfm_df = run_query("""
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        rfm_segment
    FROM dim_customer
    ORDER BY monetary DESC NULLS LAST;
""")

if not rfm_df.empty:
    rfm_col1, rfm_col2 = st.columns(2)

    with rfm_col1:
        fig_rfm = px.scatter(
            rfm_df,
            x="frequency",
            y="monetary",
            color="rfm_segment",
            hover_data=["customer_id", "recency_days"],
            title="Customer Frequency vs Monetary Value",
            labels={
                "frequency": "Purchase Frequency",
                "monetary": "Monetary Value (£)",
                "rfm_segment": "RFM Segment",
            },
        )
        st.plotly_chart(fig_rfm, use_container_width=True)

    with rfm_col2:
        segment_counts = (
            rfm_df["rfm_segment"]
            .fillna("Unclassified")
            .value_counts()
            .reset_index()
        )
        segment_counts.columns = ["rfm_segment", "customers"]

        fig_segments = px.bar(
            segment_counts,
            x="rfm_segment",
            y="customers",
            title="Customers by RFM Segment",
            labels={
                "rfm_segment": "RFM Segment",
                "customers": "Customers",
            },
        )
        st.plotly_chart(fig_segments, use_container_width=True)

    st.dataframe(rfm_df, use_container_width=True)
else:
    st.info("No customer RFM data is available.")

st.divider()

# --- Customer retention ---
st.subheader("🔁 Customer Retention")

retention_df = run_query("""
    SELECT *
    FROM fact_customer_retention
    ORDER BY customer_key;
""")

if not retention_df.empty:
    st.metric(
        "Customers in Retention Snapshot",
        f"{len(retention_df):,}",
    )
    st.dataframe(retention_df, use_container_width=True)
else:
    st.info("No customer retention data is available.")

st.caption(
    "Data source: PostgreSQL data warehouse | "
    "Filters apply to sales KPIs and sales visualizations."
)
