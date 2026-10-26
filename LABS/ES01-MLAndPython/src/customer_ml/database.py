from functools import lru_cache

from pymongo import ASCENDING, MongoClient

from customer_ml.config import DATABASE_NAME, mongo_uri

COLLECTIONS = (
    "customers",
    "orders",
    "customer_features",
    "training_labels",
    "churn_events",
    "predictions",
    "model_metadata",
)

VALIDATORS = {
    "customers": {
        "bsonType": "object",
        "required": ["customer_id", "created_at", "synthetic"],
        "properties": {
            "customer_id": {"bsonType": "string"},
            "created_at": {"bsonType": "date"},
            "synthetic": {"bsonType": "bool"},
        },
    },
    "orders": {
        "bsonType": "object",
        "required": ["customer_id", "occurred_at", "status", "amount", "synthetic"],
        "properties": {
            "customer_id": {"bsonType": "string"},
            "occurred_at": {"bsonType": "date"},
            "status": {"bsonType": "string"},
            "amount": {"bsonType": ["double", "int", "long"]},
            "synthetic": {"bsonType": "bool"},
        },
    },
    "customer_features": {
        "bsonType": "object",
        "required": [
            "snapshot_id",
            "customer_id",
            "observation_at",
            "feature_schema_version",
            "features",
            "synthetic",
        ],
        "properties": {
            "snapshot_id": {"bsonType": "string"},
            "customer_id": {"bsonType": "string"},
            "observation_at": {"bsonType": "date"},
            "feature_schema_version": {"bsonType": "string"},
            "synthetic": {"bsonType": "bool"},
            "features": {
                "bsonType": "object",
                "required": [
                    "age_years",
                    "orders_365d",
                    "days_since_last_order",
                    "avg_order_value",
                    "support_tickets_90d",
                    "plan_tier",
                ],
                "properties": {
                    "age_years": {"bsonType": ["double", "int", "long"]},
                    "orders_365d": {"bsonType": ["int", "long"]},
                    "days_since_last_order": {"bsonType": ["int", "long"]},
                    "avg_order_value": {
                        "bsonType": ["double", "int", "long", "null"]
                    },
                    "support_tickets_90d": {"bsonType": ["int", "long"]},
                    "plan_tier": {"enum": ["basic", "plus", "premium"]},
                },
            },
        },
    },
    "training_labels": {
        "bsonType": "object",
        "required": [
            "snapshot_id",
            "customer_id",
            "target",
            "label",
            "observation_at",
            "outcome_start",
            "outcome_end",
            "available_at",
            "synthetic",
        ],
        "properties": {
            "snapshot_id": {"bsonType": "string"},
            "customer_id": {"bsonType": "string"},
            "target": {"enum": ["purchase", "churn"]},
            "label": {"enum": [0, 1]},
            "observation_at": {"bsonType": "date"},
            "outcome_start": {"bsonType": "date"},
            "outcome_end": {"bsonType": "date"},
            "available_at": {"bsonType": "date"},
            "synthetic": {"bsonType": "bool"},
        },
    },
    "churn_events": {
        "bsonType": "object",
        "required": ["customer_id", "occurred_at", "event_type", "synthetic"],
        "properties": {
            "customer_id": {"bsonType": "string"},
            "occurred_at": {"bsonType": "date"},
            "event_type": {"enum": ["synthetic_churn"]},
            "synthetic": {"bsonType": "bool"},
        },
    },
    "predictions": {
        "bsonType": "object",
        "required": [
            "customer_id",
            "target",
            "model_version",
            "predicted_class",
            "created_at",
            "synthetic",
        ],
        "properties": {
            "customer_id": {"bsonType": "string"},
            "target": {"enum": ["purchase", "churn"]},
            "model_version": {"bsonType": "string"},
            "predicted_class": {"enum": [0, 1]},
            "probability": {
                "bsonType": ["double", "int", "long", "null"]
            },
            "created_at": {"bsonType": "date"},
            "synthetic": {"bsonType": "bool"},
        },
    },
    "model_metadata": {
        "bsonType": "object",
        "required": [
            "target",
            "model_version",
            "feature_schema_version",
            "trained_at",
            "metrics",
            "synthetic_only",
            "approved_for_demo_inference",
        ],
        "properties": {
            "target": {"enum": ["purchase", "churn"]},
            "model_version": {"bsonType": "string"},
            "feature_schema_version": {"bsonType": "string"},
            "trained_at": {"bsonType": "date"},
            "metrics": {"bsonType": "object"},
            "synthetic_only": {"bsonType": "bool"},
            "approved_for_demo_inference": {"bsonType": "bool"},
        },
    },
}


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(mongo_uri(), serverSelectionTimeoutMS=3000)


def get_database():
    return get_client()[DATABASE_NAME]


def initialize_database(database=None) -> None:
    if database is None:
        database = get_database()
    for name in COLLECTIONS:
        if name not in database.list_collection_names():
            database.create_collection(
                name, validator={"$jsonSchema": VALIDATORS[name]}
            )

    database.customers.create_index([("customer_id", ASCENDING)], unique=True)
    database.orders.create_index(
        [("customer_id", ASCENDING), ("occurred_at", ASCENDING)]
    )
    database.customer_features.create_index(
        [("customer_id", ASCENDING), ("observation_at", ASCENDING)], unique=True
    )
    database.training_labels.create_index(
        [("snapshot_id", ASCENDING), ("target", ASCENDING)], unique=True
    )
    database.churn_events.create_index(
        [("customer_id", ASCENDING), ("occurred_at", ASCENDING)]
    )
    database.predictions.create_index(
        [("customer_id", ASCENDING), ("created_at", ASCENDING)]
    )
    database.model_metadata.create_index(
        [("target", ASCENDING), ("trained_at", ASCENDING)]
    )
