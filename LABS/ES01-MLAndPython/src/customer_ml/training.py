from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_ml.config import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, TARGETS
from customer_ml.synthetic import rows_for_target

NUMERIC_FEATURES = FEATURE_COLUMNS[:-1]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[-1:]


def _split_chronologically(frame: pd.DataFrame):
    dates = sorted(frame["observation_at"].unique())
    if len(dates) < 6:
        raise ValueError("At least six observation periods are required")
    first_cut = max(1, int(len(dates) * 0.70))
    second_cut = max(first_cut + 1, int(len(dates) * 0.85))
    second_cut = min(second_cut, len(dates) - 1)
    train_dates = set(dates[:first_cut])
    validation_dates = set(dates[first_cut:second_cut])
    test_dates = set(dates[second_cut:])
    train = frame[frame["observation_at"].isin(train_dates)]
    validation = frame[frame["observation_at"].isin(validation_dates)]
    test = frame[frame["observation_at"].isin(test_dates)]
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        if subset.empty or subset["label"].nunique() != 2:
            raise ValueError(
                f"{name} split must contain both target classes; "
                "provide more observations or adjust synthetic data generation"
            )
    return train, validation, test


def _pipeline(seed: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )


def train_target(
    database,
    target: str,
    model_dir: str | Path,
    seed: int = 20261025,
) -> dict:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    rows = rows_for_target(database, target)
    if not rows:
        raise ValueError(f"No synthetic training labels found for {target}")
    if any(row["feature_schema_version"] != FEATURE_SCHEMA_VERSION for row in rows):
        raise ValueError("Training data contains an unsupported feature schema")
    if any(
        row["outcome_start"] != row["observation_at"]
        or row["available_at"] < row["outcome_end"]
        for row in rows
    ):
        raise ValueError("A training label has inconsistent observation/outcome dates")

    frame = pd.DataFrame(
        [
            {**row["features"], "label": int(row["label"]), "observation_at": row["observation_at"]}
            for row in rows
        ]
    )
    train, validation, test = _split_chronologically(frame)
    estimator = _pipeline(seed)
    estimator.fit(train[list(FEATURE_COLUMNS)], train["label"])

    expected = test["label"].to_numpy(dtype=int)
    predicted = estimator.predict(test[list(FEATURE_COLUMNS)]).astype(int)
    probabilities = estimator.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]
    metrics = {
        "precision": float(precision_score(expected, predicted, zero_division=0)),
        "recall": float(recall_score(expected, predicted, zero_division=0)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(expected, probabilities)),
        "average_precision": float(average_precision_score(expected, probabilities)),
        "positive_class": 1,
        "decision_threshold": 0.5,
        "threshold_note": "scikit-learn classifier default; not a business operating threshold",
    }

    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("%Y%m%dT%H%M%SZ")
    dataset_rows = [
        {
            "snapshot_id": row["snapshot_id"],
            "label": int(row["label"]),
            "observation_at": row["observation_at"].isoformat(),
            "features": row["features"],
        }
        for row in sorted(rows, key=lambda row: row["snapshot_id"])
    ]
    dataset_hash = hashlib.sha256(
        json.dumps(
            dataset_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact = {
        "target": target,
        "model_version": version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "estimator": estimator,
        "metrics": metrics,
    }
    destination = Path(model_dir) / f"{target}.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    joblib.dump(artifact, temporary)
    os.replace(temporary, destination)

    metadata = {
        "_id": f"{target}:{version}",
        "target": target,
        "model_version": version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_at": timestamp,
        "seed": seed,
        "dataset_sha256": dataset_hash,
        "parameters": {
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "split": "chronological 70/15/15 by observation period",
        },
        "sample_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "metrics": metrics,
        "synthetic_only": True,
        "artifact": str(destination),
        "approved_for_demo_inference": True,
    }
    database.model_metadata.replace_one(
        {"_id": metadata["_id"]}, metadata, upsert=True
    )
    return metadata
