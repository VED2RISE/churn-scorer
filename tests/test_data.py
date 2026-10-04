import pandas as pd
import pytest

from churn.data import load_events, load_labels


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_events_parses_utc_and_fills_amount(tmp_path):
    path = _write(
        tmp_path,
        "events.csv",
        "event_id,user_id,ts,event_type,amount\n"
        "e1,u1,2025-05-01T10:00:00Z,session,\n"
        "e2,u1,2025-05-01T10:05:00Z,purchase,12.5\n",
    )
    df = load_events(path)
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["amount"].tolist() == [0.0, 12.5]


def test_load_events_missing_column_raises(tmp_path):
    path = _write(tmp_path, "events.csv", "event_id,user_id,ts\ne1,u1,2025-05-01T10:00:00Z\n")
    with pytest.raises(ValueError, match="отсутствуют колонки"):
        load_events(path)


def test_load_events_unknown_type_raises(tmp_path):
    path = _write(
        tmp_path, "events.csv", "event_id,user_id,ts,event_type,amount\ne1,u1,2025-05-01T10:00:00Z,login,\n"
    )
    with pytest.raises(ValueError, match="неизвестные типы"):
        load_events(path)


def test_load_labels_rejects_duplicates(tmp_path):
    path = _write(tmp_path, "labels.csv", "user_id,churned\nu1,0\nu1,1\n")
    with pytest.raises(ValueError, match="повторяющиеся"):
        load_labels(path)


def test_load_labels_casts_churned_to_int(tmp_path):
    path = _write(tmp_path, "labels.csv", "user_id,churned\nu1,0\nu2,1\n")
    df = load_labels(path)
    assert df["churned"].tolist() == [0, 1]
    assert pd.api.types.is_integer_dtype(df["churned"])
