from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_raw_data(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    users = pd.read_csv(raw_dir / "users.csv", parse_dates=["register_time"])
    events = pd.read_csv(raw_dir / "events.csv", parse_dates=["event_time"])
    orders = pd.read_csv(raw_dir / "orders.csv", parse_dates=["order_time"])
    required = {
        "users": {"user_id", "register_time"},
        "events": {"user_id", "event_time", "event_type"},
        "orders": {"user_id", "order_time", "amount", "order_status"},
    }
    for name, frame in [("users", users), ("events", events), ("orders", orders)]:
        missing = required[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv 缺少字段: {sorted(missing)}")
    return users, events, orders


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
