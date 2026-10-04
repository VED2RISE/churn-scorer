"""Построение признаков пользователя на дату cutoff."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from churn.config import NO_SESSION_DAYS, WINDOW_DAYS

FEATURE_COLUMNS = [
    "n_sessions_30d",
    "n_purchases_30d",
    "spend_30d",
    "n_tickets_30d",
    "days_since_last_session",
    "tenure_days",
    "sessions_per_week_lifetime",
]


def _recent(events: pd.DataFrame, cutoff: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """События за последние `window_days` дней перед cutoff."""
    start = cutoff - pd.Timedelta(days=window_days)
    return events[events["ts"] >= start]


def build_features(
    events: pd.DataFrame,
    user_ids: Iterable[str],
    cutoff: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Признаки с индексом user_id — по строке на каждого из `user_ids`.

    Признаки описывают состояние пользователя на момент `cutoff`.
    Пользователи без событий получают нули (и NO_SESSION_DAYS в days_since_last_session).
    """
    index = pd.Index(pd.unique(pd.Series(list(user_ids))), name="user_id")
    feats = pd.DataFrame(index=index)

    recent = _recent(events, cutoff, window_days)
    counts = recent.pivot_table(
        index="user_id", columns="event_type", values="event_id", aggfunc="count", fill_value=0
    ).reindex(index, fill_value=0)
    feats["n_sessions_30d"] = counts["session"] if "session" in counts else 0
    feats["n_purchases_30d"] = counts["purchase"] if "purchase" in counts else 0
    feats["n_tickets_30d"] = counts["support_ticket"] if "support_ticket" in counts else 0

    purchases = recent[recent["event_type"] == "purchase"]
    feats["spend_30d"] = purchases.groupby("user_id")["amount"].sum().reindex(index, fill_value=0.0)

    sessions = events[events["event_type"] == "session"]
    last_session = sessions.groupby("user_id")["ts"].max().reindex(index)
    feats["days_since_last_session"] = (cutoff - last_session).dt.days.fillna(NO_SESSION_DAYS)

    first_event = events.groupby("user_id")["ts"].min().reindex(index)
    feats["tenure_days"] = (cutoff - first_event).dt.days.fillna(0)

    lifetime_sessions = sessions.groupby("user_id").size().reindex(index, fill_value=0)
    weeks = feats["tenure_days"].clip(lower=1) / 7.0
    feats["sessions_per_week_lifetime"] = lifetime_sessions / weeks

    return feats[FEATURE_COLUMNS].astype(float)
