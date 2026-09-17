from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "recency_days", "frequency_90d", "monetary_90d", "events_30d", "events_90d",
    "clicks_30d", "browses_30d", "carts_30d", "purchases_30d", "active_days_30d",
    "avg_order_amount_90d", "days_since_register", "city", "channel",
]


def build_features(users, events, orders, observation_end, label_window_days=30):
    """构造无未来信息泄漏的用户特征及未来窗口流失标签。"""
    end = pd.Timestamp(observation_end)
    label_end = end + pd.Timedelta(days=label_window_days)
    users, events, orders = users.copy(), events.copy(), orders.copy()
    users["register_time"] = pd.to_datetime(users["register_time"])
    events["event_time"] = pd.to_datetime(events["event_time"])
    orders["order_time"] = pd.to_datetime(orders["order_time"])
    base = users[["user_id", "register_time", "city", "channel"]].drop_duplicates("user_id")
    e90 = events[(events.event_time > end - pd.Timedelta(days=90)) & (events.event_time <= end)]
    e30 = e90[e90.event_time > end - pd.Timedelta(days=30)]
    paid = orders[orders.order_status.eq("paid")]
    o90 = paid[(paid.order_time > end - pd.Timedelta(days=90)) & (paid.order_time <= end)]

    event_agg = e90.groupby("user_id").agg(events_90d=("event_type", "size"), last_event_time=("event_time", "max"))
    event30_agg = e30.groupby("user_id").agg(events_30d=("event_type", "size"), active_days_30d=("event_time", lambda x: x.dt.date.nunique()))
    type_counts = pd.crosstab(e30["user_id"], e30["event_type"]).rename(columns={"click": "clicks_30d", "browse": "browses_30d", "cart": "carts_30d", "purchase": "purchases_30d"})
    order_agg = o90.groupby("user_id").agg(frequency_90d=("order_time", "size"), monetary_90d=("amount", "sum"), avg_order_amount_90d=("amount", "mean"), last_order_time=("order_time", "max"))
    features = base.join(event_agg, on="user_id").join(event30_agg, on="user_id").join(type_counts, on="user_id").join(order_agg, on="user_id")
    features["last_activity_time"] = features[["last_event_time", "last_order_time"]].max(axis=1)
    features["recency_days"] = (end - features["last_activity_time"]).dt.days
    features["days_since_register"] = (end - features["register_time"]).dt.days.clip(lower=0)
    future_events = events[(events.event_time > end) & (events.event_time <= label_end)]
    future_orders = paid[(paid.order_time > end) & (paid.order_time <= label_end)]
    future_ids = set(future_events.user_id) | set(future_orders.user_id)
    features["churned"] = (~features.user_id.isin(future_ids)).astype(int)
    features = features.drop(columns=["register_time", "last_event_time", "last_order_time", "last_activity_time"], errors="ignore")
    numeric = [c for c in FEATURE_COLUMNS if c not in {"city", "channel"}]
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)
    features["recency_days"] = features["recency_days"].replace(0, 999).clip(upper=999)
    for column in ["city", "channel"]:
        features[column] = features[column].fillna("unknown").astype(str)
    return features[FEATURE_COLUMNS + ["user_id", "churned"]]


def validate_features(frame):
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"特征数据缺少字段: {sorted(missing)}")
