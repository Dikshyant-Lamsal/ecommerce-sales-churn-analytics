"""Reproducible customer snapshot features and forward-looking churn labels.

Churn definition: a customer is churned at a monthly snapshot if they make no
qualifying purchase in the following 90 days. Features use only the 180 days
strictly before the snapshot. This avoids using future transactions as inputs.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

FEATURE_COLUMNS = [
    "recency_days", "orders_180d", "monetary_180d", "avg_order_value_180d",
    "items_180d", "tenure_days", "unique_products_180d",
]
CHURN_WINDOW_DAYS = 90
FEATURE_WINDOW_DAYS = 180


def load_transactions(path: str | None = None) -> pd.DataFrame:
    """Load cleaned CSV, raw CSV, or the UCI Excel file and normalize columns."""
    candidates = [Path(path)] if path else [
        Path("data/staging/cleaned_transactions.csv"),
        Path("data/landing/Online Retail.xlsx"),
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "No transaction dataset found. Place the UCI workbook at "
            "data/landing/Online Retail.xlsx or create data/staging/cleaned_transactions.csv."
        )
    df = pd.read_excel(source) if source.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(source)
    aliases = {
        "InvoiceNo": "invoice_no", "Invoice": "invoice_no", "StockCode": "stock_code",
        "Description": "description", "Quantity": "quantity", "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price", "Price": "unit_price", "CustomerID": "customer_id", "Country": "country",
    }
    df = df.rename(columns={c: aliases.get(c, c.lower()) for c in df.columns})
    required = {"invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df = df.dropna(subset=["invoice_date", "quantity", "unit_price", "customer_id"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0) & (~df["invoice_no"].str.startswith("C"))]
    df = df[df["customer_id"] != -1].copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["stock_code"] = df["stock_code"].astype(str)
    return df.sort_values("invoice_date").reset_index(drop=True)


def build_customer_snapshots(transactions: pd.DataFrame, churn_days: int = CHURN_WINDOW_DAYS,
                             feature_days: int = FEATURE_WINDOW_DAYS) -> pd.DataFrame:
    """Create monthly customer snapshots with historical features and 90-day labels."""
    df = transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["revenue"] if "revenue" in df else df["quantity"] * df["unit_price"]
    first_date, last_date = df["invoice_date"].min().normalize(), df["invoice_date"].max().normalize()
    # Only snapshots with a complete forward label window are eligible.
    last_snapshot = last_date - pd.Timedelta(days=churn_days)
    if last_snapshot <= first_date:
        raise ValueError("Not enough transaction history to create a complete churn-label window.")
    snapshots = pd.date_range(first_date + pd.offsets.MonthEnd(0), last_snapshot, freq="ME")
    # Handle datasets whose first month-end is already past the last eligible snapshot.
    if len(snapshots) == 0:
        snapshots = pd.DatetimeIndex([last_snapshot])
    rows: list[dict] = []
    for snapshot in snapshots:
        hist_start = snapshot - pd.Timedelta(days=feature_days)
        hist = df[(df.invoice_date <= snapshot) & (df.invoice_date > hist_start)]
        future = df[(df.invoice_date > snapshot) & (df.invoice_date <= snapshot + pd.Timedelta(days=churn_days))]
        future_customers = set(future.customer_id.unique())
        if hist.empty:
            continue
        first_purchase_dates = df.groupby("customer_id")["invoice_date"].min()
        for cid, group in hist.groupby("customer_id", sort=False):
            last_purchase = group.invoice_date.max()
            future_bought = cid in future_customers
            orders = int(group.invoice_no.nunique())
            monetary = float(group.revenue.sum())
            rows.append({
                "customer_id": int(cid), "snapshot_date": snapshot.date().isoformat(),
                "recency_days": int((snapshot - last_purchase.normalize()).days),
                "orders_180d": orders, "monetary_180d": round(monetary, 2),
                "avg_order_value_180d": round(monetary / max(orders, 1), 2),
                "items_180d": int(group.quantity.sum()), "tenure_days": int((snapshot - first_purchase_dates.loc[cid].normalize()).days),
                "unique_products_180d": int(group.stock_code.nunique()), "is_churned": int(not future_bought),
            })
    result = pd.DataFrame(rows)
    if result.empty or result.is_churned.nunique() < 2:
        raise ValueError("Snapshot dataset has insufficient rows or only one churn class. Check dataset dates/window.")
    return result


def make_training_data(path: str | None = None) -> pd.DataFrame:
    return build_customer_snapshots(load_transactions(path))
