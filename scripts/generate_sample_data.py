from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main(n_users=1200, seed=42):
    rng = np.random.default_rng(seed); OUT.mkdir(parents=True, exist_ok=True)
    users = pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(n_users)], "register_time": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 450, n_users), unit="D"), "city": rng.choice(["Shanghai", "Beijing", "Shenzhen", "Guangzhou", "unknown"], n_users, p=[.25, .22, .2, .18, .15]), "channel": rng.choice(["organic", "ads", "referral", "social"], n_users)})
    event_users = rng.choice(users.user_id, 24000); events = pd.DataFrame({"user_id": event_users, "event_time": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 580, 24000), unit="D"), "event_type": rng.choice(["click", "browse", "cart", "purchase"], 24000, p=[.35, .4, .15, .1])})
    order_users = rng.choice(users.user_id, 8000); orders = pd.DataFrame({"user_id": order_users, "order_time": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 580, 8000), unit="D"), "amount": np.round(rng.gamma(2.2, 85, 8000), 2), "order_status": np.where(rng.random(8000) < .7, "paid", "cancelled")})
    users.to_csv(OUT / "users.csv", index=False); events.to_csv(OUT / "events.csv", index=False); orders.to_csv(OUT / "orders.csv", index=False)
    print(f"Generated {n_users} users, {len(events)} events and {len(orders)} orders in {OUT}")


if __name__ == "__main__": main()
