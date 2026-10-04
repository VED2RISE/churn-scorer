"""Пути и константы проекта."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

# Исторические данные для обучения: полная выгрузка таблицы событий.
TRAIN_EVENTS = DATA_DIR / "train_events.csv"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
TRAIN_CUTOFF = pd.Timestamp("2025-06-01", tz="UTC")

# Holdout: снапшот событий на дату пилота, метки собраны через 30 дней.
HOLDOUT_EVENTS = DATA_DIR / "holdout_events.csv"
HOLDOUT_LABELS = DATA_DIR / "holdout_labels.csv"
HOLDOUT_CUTOFF = pd.Timestamp("2025-08-01", tz="UTC")

WINDOW_DAYS = 30  # окно "недавней" активности для признаков *_30d
LABEL_HORIZON_DAYS = 30  # churned = ни одной сессии в течение 30 дней после cutoff
NO_SESSION_DAYS = 365  # days_since_last_session для пользователей без сессий

RANDOM_STATE = 42
TEST_SIZE = 0.25
