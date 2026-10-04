import pandas as pd

from churn.config import NO_SESSION_DAYS
from churn.features import FEATURE_COLUMNS, build_features

CUTOFF = pd.Timestamp("2025-06-01", tz="UTC")


def _events(rows):
    """rows: (user_id, ts, event_type, amount)."""
    df = pd.DataFrame(rows, columns=["user_id", "ts", "event_type", "amount"])
    df.insert(0, "event_id", [f"e{i}" for i in range(len(df))])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["amount"] = df["amount"].astype(float).fillna(0.0)
    return df


def test_recent_window_counts_and_spend():
    events = _events(
        [
            ("u1", "2025-04-10T12:00:00Z", "session", None),  # 52 дня до cutoff — вне окна
            ("u1", "2025-05-10T12:00:00Z", "session", None),
            ("u1", "2025-05-20T12:00:00Z", "session", None),
            ("u1", "2025-05-20T12:03:00Z", "purchase", 25.0),
            ("u1", "2025-05-25T09:00:00Z", "support_ticket", None),
            ("u1", "2025-05-29T12:00:00Z", "session", None),
        ]
    )
    feats = build_features(events, ["u1"], CUTOFF)
    row = feats.loc["u1"]
    assert row["n_sessions_30d"] == 3
    assert row["n_purchases_30d"] == 1
    assert row["spend_30d"] == 25.0
    assert row["n_tickets_30d"] == 1


def test_days_since_last_session_and_tenure():
    events = _events(
        [
            ("u1", "2025-02-21T00:00:00Z", "session", None),  # первый ивент: ровно 100 дней до cutoff
            ("u1", "2025-05-29T00:00:00Z", "session", None),  # последняя сессия: ровно 3 дня до cutoff
        ]
    )
    feats = build_features(events, ["u1"], CUTOFF)
    assert feats.loc["u1", "days_since_last_session"] == 3
    assert feats.loc["u1", "tenure_days"] == 100


def test_user_without_events_gets_defaults():
    events = _events([("u1", "2025-05-29T12:00:00Z", "session", None)])
    feats = build_features(events, ["u1", "ghost"], CUTOFF)
    ghost = feats.loc["ghost"]
    assert ghost["n_sessions_30d"] == 0
    assert ghost["spend_30d"] == 0
    assert ghost["days_since_last_session"] == NO_SESSION_DAYS
    assert ghost["tenure_days"] == 0
    assert ghost["sessions_per_week_lifetime"] == 0


def test_output_shape_and_columns():
    events = _events([("u1", "2025-05-29T12:00:00Z", "session", None)])
    feats = build_features(events, ["u1", "u2", "u1"], CUTOFF)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert list(feats.index) == ["u1", "u2"]
    assert feats.index.name == "user_id"
