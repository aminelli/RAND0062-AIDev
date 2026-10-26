import pandas as pd
import pytest

from customer_ml.config import FEATURE_COLUMNS
from customer_ml.synthetic import make_synthetic_documents
from customer_ml.training import _pipeline, _split_chronologically


def test_split_is_chronological_and_keeps_classes():
    periods = pd.date_range("2023-01-01", periods=10, freq="MS")
    frame = pd.DataFrame(
        {
            "observation_at": list(periods.repeat(20)),
            "label": [index % 2 for index in range(len(periods) * 20)],
        }
    )

    train, validation, test = _split_chronologically(frame)

    assert train["observation_at"].max() < validation["observation_at"].min()
    assert validation["observation_at"].max() < test["observation_at"].min()
    assert all(part["label"].nunique() == 2 for part in (train, validation, test))


def test_split_rejects_single_class_period_part():
    frame = pd.DataFrame(
        {
            "observation_at": pd.date_range("2023-01-01", periods=8, freq="MS"),
            "label": [0] * 8,
        }
    )

    with pytest.raises(ValueError, match="both target classes"):
        _split_chronologically(frame)


def test_preprocessing_and_classifier_fit_on_synthetic_rows():
    docs = make_synthetic_documents(count=720, seed=20261025)
    features_by_id = {row["snapshot_id"]: row for row in docs["customer_features"]}
    rows = [
        {
            **features_by_id[label["snapshot_id"]]["features"],
            "label": label["label"],
            "observation_at": label["observation_at"],
        }
        for label in docs["training_labels"]
        if label["target"] == "purchase"
    ]
    frame = pd.DataFrame(rows)
    train, _, test = _split_chronologically(frame)
    estimator = _pipeline(seed=20261025)
    estimator.fit(train[list(FEATURE_COLUMNS)], train["label"])

    probabilities = estimator.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]

    assert len(probabilities) == len(test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
