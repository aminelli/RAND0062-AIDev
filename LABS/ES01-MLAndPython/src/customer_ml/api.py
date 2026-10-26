from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pymongo.errors import PyMongoError

from customer_ml.config import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, MODEL_DIR, TARGETS
from customer_ml.database import get_database


def _load_artifact(path: Path, target: str) -> dict:
    try:
        artifact = joblib.load(path)
    except (OSError, ValueError, EOFError) as exc:
        raise HTTPException(status_code=503, detail="Model artifact is invalid") from exc
    if (
        not isinstance(artifact, dict)
        or "estimator" not in artifact
        or "model_version" not in artifact
        or artifact.get("target") != target
        or artifact.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or artifact.get("feature_columns") != list(FEATURE_COLUMNS)
    ):
        raise HTTPException(
            status_code=503, detail="Model artifact is incompatible with current features"
        )
    return artifact


class PredictionResponse(BaseModel):
    customer_id: str
    target: str
    model_version: str
    predicted_class: int
    probability: float | None
    created_at: datetime
    synthetic: bool


class FeatureVector(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    age_years: float
    orders_365d: int
    days_since_last_order: int
    avg_order_value: float | None
    support_tickets_90d: int
    plan_tier: Literal["basic", "plus", "premium"]


def create_app(database_provider=get_database, model_dir: str | Path = MODEL_DIR):
    application = FastAPI(
        title="Customer Purchase & Churn Demo",
        version="0.1.0",
        description="Local demonstration using synthetic data only.",
    )

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/health/database")
    def database_health():
        try:
            database_provider().command("ping")
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        return {"status": "ok"}

    @application.get("/customers")
    def customers(limit: int = Query(default=100, ge=1, le=500)):
        try:
            return list(
                database_provider().customers.find(
                    {"synthetic": True}, {"_id": 0, "customer_id": 1}
                ).sort("customer_id", 1).limit(limit)
            )
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc

    @application.get("/customers/{customer_id}")
    def customer_features(customer_id: str):
        try:
            snapshot = database_provider().customer_features.find_one(
                {"customer_id": customer_id, "synthetic": True},
                {"_id": 0},
                sort=[("observation_at", -1)],
            )
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Customer snapshot not found")
        return snapshot

    @application.get("/models")
    def models():
        try:
            database = database_provider()
            output = []
            for target in TARGETS:
                record = database.model_metadata.find_one(
                    {"target": target, "approved_for_demo_inference": True},
                    {"_id": 0},
                    sort=[("trained_at", -1)],
                )
                artifact_path = Path(model_dir) / f"{target}.joblib"
                available = bool(record and artifact_path.is_file())
                unavailable_reason = None
                if not record:
                    unavailable_reason = "Validated model metadata is not available"
                elif not artifact_path.is_file():
                    unavailable_reason = "Model artifact file is not available"
                else:
                    try:
                        artifact = _load_artifact(artifact_path, target)
                        available = artifact["model_version"] == record["model_version"]
                        if not available:
                            unavailable_reason = "Artifact and metadata versions do not match"
                    except HTTPException as exc:
                        available = False
                        unavailable_reason = exc.detail
                output.append(
                    {
                        "target": target,
                        "available": available,
                        "unavailable_reason": unavailable_reason,
                        "metadata": record,
                    }
                )
            return output
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc

    @application.get("/metrics")
    def metrics():
        try:
            database = database_provider()
            return [
                {
                    "target": target,
                    "metadata": database.model_metadata.find_one(
                        {"target": target, "approved_for_demo_inference": True},
                        {"_id": 0},
                        sort=[("trained_at", -1)],
                    ),
                }
                for target in TARGETS
            ]
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc

    @application.post(
        "/predictions/{target}/{customer_id}", response_model=PredictionResponse
    )
    def predict(target: str, customer_id: str):
        if target not in TARGETS:
            raise HTTPException(status_code=404, detail="Unknown prediction target")
        artifact_path = Path(model_dir) / f"{target}.joblib"
        if not artifact_path.is_file():
            raise HTTPException(
                status_code=503, detail=f"Validated {target} model is unavailable"
            )
        artifact = _load_artifact(artifact_path, target)
        try:
            database = database_provider()
            metadata = database.model_metadata.find_one(
                {
                    "_id": f"{target}:{artifact['model_version']}",
                    "approved_for_demo_inference": True,
                }
            )
            if metadata is None:
                raise HTTPException(
                    status_code=503, detail="Model is not approved for demo inference"
                )
            snapshot = database.customer_features.find_one(
                {"customer_id": customer_id, "synthetic": True},
                {"_id": 0},
                sort=[("observation_at", -1)],
            )
            if snapshot is None:
                raise HTTPException(status_code=404, detail="Customer snapshot not found")
            if snapshot.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                raise HTTPException(
                    status_code=503, detail="Customer feature schema is incompatible"
                )
            try:
                values = FeatureVector.model_validate(snapshot["features"]).model_dump()
            except (KeyError, ValueError) as exc:
                raise HTTPException(
                    status_code=503, detail="Customer features are invalid"
                ) from exc
            input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            predicted = int(artifact["estimator"].predict(input_frame)[0])
            probabilities = getattr(artifact["estimator"], "predict_proba", None)
            probability = (
                float(probabilities(input_frame)[0][1]) if probabilities else None
            )
            created_at = datetime.now(timezone.utc)
            result = {
                "customer_id": customer_id,
                "target": target,
                "model_version": artifact["model_version"],
                "predicted_class": predicted,
                "probability": probability,
                "created_at": created_at,
                "synthetic": True,
            }
            database.predictions.insert_one(result)
            return result
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database operation failed") from exc

    @application.get("/predictions")
    def prediction_history(
        target: str | None = None, limit: int = Query(default=100, ge=1, le=500)
    ):
        if target is not None and target not in TARGETS:
            raise HTTPException(status_code=400, detail="Unknown prediction target")
        query = {"synthetic": True}
        if target:
            query["target"] = target
        try:
            return list(
                database_provider().predictions.find(
                    query, {"_id": 0}
                ).sort("created_at", -1).limit(limit)
            )
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Database unavailable") from exc

    return application


app = create_app()
