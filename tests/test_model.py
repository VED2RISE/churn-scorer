import numpy as np
import pandas as pd

from churn.features import FEATURE_COLUMNS
from churn.model import ChurnModel, evaluate, train


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.uniform(0, 10, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    feats.index = pd.Index([f"u{i}" for i in range(n)], name="user_id")
    logit = 1.5 - 0.4 * feats["n_sessions_30d"] + 0.3 * feats["days_since_last_session"] + rng.normal(0, 1, n)
    labels = pd.Series((logit > 0).astype(int), index=feats.index, name="churned")
    return feats, labels


def test_train_returns_sane_metrics():
    feats, labels = _synthetic()
    model, metrics = train(feats, labels)
    assert metrics["n_train"] + metrics["n_val"] == len(feats)
    assert 0.5 < metrics["val_auc"] <= 1.0
    assert model.feature_columns == FEATURE_COLUMNS


def test_predict_proba_range_and_shape():
    feats, labels = _synthetic()
    model, _ = train(feats, labels)
    proba = model.predict_proba(feats)
    assert proba.shape == (len(feats),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_save_load_roundtrip(tmp_path):
    feats, labels = _synthetic()
    model, _ = train(feats, labels)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = ChurnModel.load(path)
    assert np.allclose(loaded.predict_proba(feats), model.predict_proba(feats))


def test_evaluate_on_training_data():
    feats, labels = _synthetic()
    model, _ = train(feats, labels)
    metrics = evaluate(model, feats, labels)
    assert metrics["n"] == len(feats)
    assert 0.5 < metrics["auc"] <= 1.0
