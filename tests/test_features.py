import pandas as pd
from churn_prediction.features import FEATURE_COLUMNS, build_features


def test_features_are_leakage_safe_and_have_expected_columns():
    users = pd.DataFrame({"user_id": ["u1", "u2"], "register_time": ["2025-01-01", "2025-01-01"], "city": ["A", "B"], "channel": ["organic", "ads"]})
    events = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "event_time": ["2025-06-20", "2025-07-10", "2025-08-01"], "event_type": ["browse", "click", "browse"]})
    orders = pd.DataFrame({"user_id": ["u1"], "order_time": ["2025-06-10"], "amount": [100.0], "order_status": ["paid"]})
    result = build_features(users, events, orders, "2025-06-30", 30)
    assert set(FEATURE_COLUMNS + ["user_id", "churned"]) == set(result.columns)
    assert int(result.loc[result.user_id == "u1", "churned"].iloc[0]) == 0
    assert int(result.loc[result.user_id == "u2", "churned"].iloc[0]) == 1
