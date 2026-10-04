"""Обучение, сохранение и применение модели."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from churn.config import RANDOM_STATE, TEST_SIZE
from churn.features import FEATURE_COLUMNS


@dataclass
class ChurnModel:
    scaler: StandardScaler
    clf: LogisticRegression
    feature_columns: list[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Вероятность ухода для каждой строки `features`."""
        x = self.scaler.transform(features[self.feature_columns].to_numpy(dtype=float))
        return self.clf.predict_proba(x)[:, 1]

    def coefficients(self) -> pd.Series:
        coefs = pd.Series(self.clf.coef_[0], index=self.feature_columns)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ChurnModel":
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}. Сначала выполните `python -m churn train`.")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path}: ожидался ChurnModel, получен {type(model).__name__}")
        return model


def train(features: pd.DataFrame, labels: pd.Series) -> tuple[ChurnModel, dict[str, Any]]:
    """Обучает логистическую регрессию; возвращает модель и метрики на валидации."""
    x = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=int)

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    val_scores = clf.predict_proba(x_val)[:, 1]
    metrics: dict[str, Any] = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "churn_rate": float(y.mean()),
        "val_auc": float(roc_auc_score(y_val, val_scores)),
    }
    return ChurnModel(scaler=scaler, clf=clf, feature_columns=list(FEATURE_COLUMNS)), metrics


def evaluate(model: ChurnModel, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    """Метрики сохранённой модели на новых данных."""
    scores = model.predict_proba(features)
    y = labels.to_numpy(dtype=int)
    return {
        "n": int(len(y)),
        "churn_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, scores)),
    }
