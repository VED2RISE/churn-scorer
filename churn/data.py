"""Загрузка и базовая валидация входных данных."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVENT_COLUMNS = ["event_id", "user_id", "ts", "event_type", "amount"]
EVENT_TYPES = ("session", "purchase", "support_ticket")
LABEL_COLUMNS = ["user_id", "churned"]


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: отсутствуют колонки {missing}")


def load_events(path: str | Path) -> pd.DataFrame:
    """Читает журнал событий. `ts` приводится к UTC, пустой `amount` -> 0.0."""
    df = pd.read_csv(path)
    _require_columns(df, EVENT_COLUMNS, f"events {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["amount"] = df["amount"].astype(float).fillna(0.0)
    unknown = set(df["event_type"].unique()) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"events {path}: неизвестные типы событий {sorted(unknown)}")
    return df


def load_labels(path: str | Path) -> pd.DataFrame:
    """Читает метки: по одной строке на пользователя, churned ∈ {0, 1}."""
    df = pd.read_csv(path)
    _require_columns(df, LABEL_COLUMNS, f"labels {path}")
    if df["user_id"].duplicated().any():
        raise ValueError(f"labels {path}: повторяющиеся user_id")
    df["churned"] = df["churned"].astype(int)
    return df
