"""Текстовые отчёты для CLI."""
from __future__ import annotations

from typing import Any

import pandas as pd

from churn.model import ChurnModel


def format_metrics(title: str, metrics: dict[str, Any]) -> str:
    lines = [title]
    for key, value in metrics.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"  {key:<12} {rendered}")
    return "\n".join(lines)


def feature_summary(features: pd.DataFrame) -> str:
    desc = features.describe().T[["mean", "min", "max"]]
    return "Признаки (mean / min / max):\n" + desc.to_string(float_format=lambda v: f"{v:10.2f}")


def coefficient_table(model: ChurnModel) -> str:
    coefs = model.coefficients()
    return "Коэффициенты модели (на стандартизованных признаках):\n" + coefs.to_string(
        float_format=lambda v: f"{v:+.3f}"
    )


def top_risk_table(scores: pd.DataFrame, n: int = 10) -> str:
    return f"Топ-{n} пользователей по риску ухода:\n" + scores.head(n).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"
    )
