import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="E-Commerce Sales & Retention Dashboard", layout="wide")

@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

@st.cache_data(ttl=600)
def run_query(query, params=None):
    engine = get_engine()
    return pd.read_sql(query, engine, params=params)

st.title("📊 E-Commerce Sales Analytics & Customer Retention Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

date_bounds = run_query("SELECT MIN(full_date) AS min_d, MAX(full_date) AS max_d FROM dim_date WHERE full_date <= DATE '2011-12-09';")
min_date, max_date = date_bounds['min_d'][0], date_bounds['max_d'][0]

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)

countries_df = run_query("SELECT DISTINCT country FROM dim_geography ORDER BY country;")
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries_df['country'].tolist(),
    default=[]
)

country_filter_sql = ""
if selected_countries:
    country_list = "', '".join(selected_countries)
    country_filter_sql = f"AND g.country IN ('{country_list}')"

# --- Top-level KPIs ---
kpi_df = run_query(f"""
    SELECT
        COUNT(DISTINCT f.invoice_no) AS total_orders,
        SUM(f.line_revenue) AS total_revenue,
        COUNT(DISTINCT f.customer_key) AS total_customers
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_geography g ON f.geography_key = g.geography_key
    WHERE d.full_date BETWEEN '{start_date}' AND '{end_date}'
    {country_filter_sql};
""")

col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{int(kpi_df['total_orders'][0] or 0):,}")
col2.metric("Total Revenue", f"£{(kpi_df['total_revenue'][0] or 0):,.2f}")
col3.metric("Unique Customers", f"{int(kpi_df['total_customers'][0] or 0):,}")

st.divider()

# --- View 1: Revenue and order trends ---
st.subheader("1. Revenue & Order Trends Over Time")
trend_df = run_query(f"""
    SELECT d.full_date,
           SUM(f.line_revenue) AS revenue,
           COUNT(DISTINCT f.invoice_no) AS orders
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_geography g ON f.geography_key = g.geography_key
    WHERE d.full_date BETWEEN '{start_date}' AND '{end_date}'
    {country_filter_sql}
    GROUP BY d.full_date
    ORDER BY d.full_date;
""")
fig1 = px.line(trend_df, x='full_date', y='revenue', title="Daily Revenue Trend")
st.plotly_chart(fig1, width='stretch')

fig1b = px.line(trend_df, x='full_date', y='orders', title="Daily Order Count Trend")
st.plotly_chart(fig1b, width='stretch')

st.divider()

# --- View 2: Top products and categories ---
st.subheader("2. Top Products by Revenue")
top_products_df = run_query(f"""
    SELECT p.stock_code, p.description, SUM(f.line_revenue) AS revenue, SUM(f.quantity) AS units_sold
    FROM fact_sales f
    JOIN dim_product p ON f.product_key = p.product_key
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_geography g ON f.geography_key = g.geography_key
    WHERE d.full_date BETWEEN '{start_date}' AND '{end_date}'
    {country_filter_sql}
    GROUP BY p.stock_code, p.description
    ORDER BY revenue DESC
    LIMIT 15;
""")
fig2 = px.bar(top_products_df, x='revenue', y='description', orientation='h',
              title="Top 15 Products by Revenue", height=600)
fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig2, width='stretch')

st.divider()

# --- View 3: Customer segmentation (not date/country filtered - it's a customer-level lifetime view) ---
st.subheader("3. Customer Segmentation (RFM-based)")
segment_df = run_query("""
    SELECT customer_segment, COUNT(*) AS num_customers, ROUND(AVG(monetary),2) AS avg_monetary
    FROM dim_customer
    WHERE customer_id != -1 AND customer_segment IS NOT NULL
    GROUP BY customer_segment
    ORDER BY num_customers DESC;
""")
col_a, col_b = st.columns(2)
fig3a = px.pie(segment_df, names='customer_segment', values='num_customers', title="Customer Count by Segment")
col_a.plotly_chart(fig3a, width='stretch')

fig3b = px.bar(segment_df, x='customer_segment', y='avg_monetary', title="Avg Spend by Segment")
col_b.plotly_chart(fig3b, width='stretch')

st.divider()

# --- View 4: Country-wise sales ---
st.subheader("4. Country-wise Sales")
country_df = run_query(f"""
    SELECT g.country, SUM(f.line_revenue) AS revenue, COUNT(DISTINCT f.invoice_no) AS orders
    FROM fact_sales f
    JOIN dim_geography g ON f.geography_key = g.geography_key
    JOIN dim_date d ON f.date_key = d.date_key
    WHERE d.full_date BETWEEN '{start_date}' AND '{end_date}'
    {country_filter_sql}
    GROUP BY g.country
    ORDER BY revenue DESC
    LIMIT 15;
""")
fig4 = px.bar(country_df, x='country', y='revenue', title="Top 15 Countries by Revenue")
st.plotly_chart(fig4, width='stretch')

st.divider()

# --- View 5: Retention and repeat-purchase analysis (lifetime view, not date filtered) ---
st.subheader("5. Retention & Repeat-Purchase Analysis")
retention_df = run_query("""
    SELECT is_repeat_customer, is_churned, COUNT(*) AS num_customers
    FROM fact_customer_retention
    GROUP BY is_repeat_customer, is_churned;
""")
col_c, col_d = st.columns(2)

repeat_summary = retention_df.groupby('is_repeat_customer')['num_customers'].sum().reset_index()
repeat_summary['is_repeat_customer'] = repeat_summary['is_repeat_customer'].map({True: 'Repeat', False: 'One-time'})
fig5a = px.pie(repeat_summary, names='is_repeat_customer', values='num_customers', title="Repeat vs One-time Customers")
col_c.plotly_chart(fig5a, width='stretch')

churn_summary = retention_df.groupby('is_churned')['num_customers'].sum().reset_index()
churn_summary['is_churned'] = churn_summary['is_churned'].map({True: 'Churned (>90 days inactive)', False: 'Active'})
fig5b = px.pie(churn_summary, names='is_churned', values='num_customers', title="Churned vs Active Customers")
col_d.plotly_chart(fig5b, width='stretch')

st.divider()
st.caption("Data source: UCI Online Retail Dataset | Pipeline: Pandas → PostgreSQL Star Schema | Streamlit chosen for interactive filtering and future churn-prediction integration (Part 2) | Built for Data Engineering & MLOps Project")
