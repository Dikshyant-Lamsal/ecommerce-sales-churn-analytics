import pandas as pd
from ml.features import build_customer_snapshots


def test_snapshot_features_do_not_include_future_transactions():
    dates = pd.to_datetime([
        "2020-01-05", "2020-02-05", "2020-03-05", "2020-04-05",
        "2020-05-05", "2020-06-05", "2020-07-05", "2020-08-05",
        "2020-09-05", "2020-10-05", "2020-11-05", "2020-12-05",
    ])
    rows = []
    for i, date in enumerate(dates):
        rows.append({"invoice_no": f"I{i}", "stock_code": "A", "quantity": 1,
                     "invoice_date": date, "unit_price": 10.0, "customer_id": 1,
                     "revenue": 10.0})
    for i, date in enumerate(dates[:3]):
        rows.append({"invoice_no": f"J{i}", "stock_code": "B", "quantity": 1,
                     "invoice_date": date, "unit_price": 20.0, "customer_id": 2,
                     "revenue": 20.0})
    out = build_customer_snapshots(pd.DataFrame(rows))
    assert not out.empty
    assert set(out.is_churned.unique()).issubset({0, 1})
    assert (out.recency_days >= 0).all()
    assert (out.orders_180d >= 1).all()
