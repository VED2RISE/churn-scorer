"""CLI: python -m churn {train,evaluate,score}."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from churn import config
from churn.data import load_events, load_labels
from churn.features import build_features
from churn.model import ChurnModel, evaluate, train
from churn.report import coefficient_table, feature_summary, format_metrics, top_risk_table


def _cutoff(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _aligned_labels(labels: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    return labels.set_index("user_id")["churned"].reindex(features.index)


def cmd_train(args: argparse.Namespace) -> int:
    events = load_events(args.events)
    labels = load_labels(args.labels)
    cutoff = _cutoff(args.cutoff)

    features = build_features(events, labels["user_id"], cutoff)
    model, metrics = train(features, _aligned_labels(labels, features))
    model.save(Path(args.model))

    print(format_metrics(f"Обучение (cutoff={cutoff.date()}, events={args.events})", metrics))
    print()
    print(feature_summary(features))
    print()
    print(coefficient_table(model))
    print(f"\nМодель сохранена: {args.model}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = ChurnModel.load(Path(args.model))
    events = load_events(args.events)
    labels = load_labels(args.labels)
    cutoff = _cutoff(args.cutoff)

    features = build_features(events, labels["user_id"], cutoff)
    metrics = evaluate(model, features, _aligned_labels(labels, features))
    print(format_metrics(f"Оценка (cutoff={cutoff.date()}, events={args.events})", metrics))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    model = ChurnModel.load(Path(args.model))
    events = load_events(args.events)
    cutoff = _cutoff(args.cutoff)

    features = build_features(events, events["user_id"].unique(), cutoff)
    scores = pd.DataFrame(
        {"user_id": features.index, "churn_probability": model.predict_proba(features)}
    ).sort_values("churn_probability", ascending=False)
    scores.to_csv(args.out, index=False)

    print(top_risk_table(scores, n=args.top))
    print(f"\nСкоры сохранены: {args.out} ({len(scores)} пользователей)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m churn",
        description="churn-scorer: обучение модели риска ухода и скоринг пользователей",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="обучить модель на исторических данных и сохранить её")
    t.add_argument("--events", default=str(config.TRAIN_EVENTS))
    t.add_argument("--labels", default=str(config.TRAIN_LABELS))
    t.add_argument("--cutoff", default=config.TRAIN_CUTOFF.strftime("%Y-%m-%d"), help="дата среза, YYYY-MM-DD")
    t.add_argument("--model", default=str(config.MODEL_PATH))
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="оценить сохранённую модель на holdout-данных")
    e.add_argument("--events", default=str(config.HOLDOUT_EVENTS))
    e.add_argument("--labels", default=str(config.HOLDOUT_LABELS))
    e.add_argument("--cutoff", default=config.HOLDOUT_CUTOFF.strftime("%Y-%m-%d"), help="дата среза, YYYY-MM-DD")
    e.add_argument("--model", default=str(config.MODEL_PATH))
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("score", help="посчитать вероятность ухода для всех пользователей из events")
    s.add_argument("--events", default=str(config.HOLDOUT_EVENTS))
    s.add_argument("--cutoff", default=config.HOLDOUT_CUTOFF.strftime("%Y-%m-%d"), help="дата среза, YYYY-MM-DD")
    s.add_argument("--model", default=str(config.MODEL_PATH))
    s.add_argument("--out", default=str(config.ROOT / "scores.csv"))
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
