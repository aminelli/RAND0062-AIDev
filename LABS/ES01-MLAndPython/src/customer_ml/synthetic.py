from datetime import date, datetime, time, timedelta, timezone

import numpy as np
from pymongo import ReplaceOne

from customer_ml.config import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, TARGETS

SEED = 20261025
OUTCOME_DAYS = 30
LABEL_DELAY_DAYS = 1


def _utc(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def make_synthetic_documents(count: int = 720, seed: int = SEED):
    if count < 60:
        raise ValueError("count must be at least 60 to support temporal evaluation")

    rng = np.random.default_rng(seed)
    collections = {
        name: []
        for name in (
            "customers",
            "orders",
            "customer_features",
            "training_labels",
            "churn_events",
        )
    }
    for index in range(count):
        month = index % 24
        cohort_year = 2023 + month // 12
        cohort_month = month % 12 + 1
        observation_date = date(cohort_year, cohort_month, 1)
        observation_at = _utc(observation_date)
        customer_id = f"synthetic-{index:06d}"
        snapshot_id = f"{customer_id}:{observation_date.isoformat()}"

        plan = str(rng.choice(["basic", "plus", "premium"], p=[0.55, 0.30, 0.15]))
        age_years = round(float(rng.uniform(0.25, 10.0)), 2)
        history_days = max(1, min(365, int(age_years * 365)))
        orders_365d = int(rng.poisson(4.0 * history_days / 365))
        days_since_last_order = (
            int(rng.integers(1, history_days + 1))
            if orders_365d
            else int(rng.integers(history_days + 1, history_days + 366))
        )
        support_tickets = int(rng.poisson(1.2))

        collections["customers"].append(
            {
                "_id": customer_id,
                "customer_id": customer_id,
                "synthetic": True,
                "created_at": observation_at - timedelta(days=int(age_years * 365)),
            }
        )
        historical_amounts = []
        for order_index in range(orders_365d):
            elapsed = (
                days_since_last_order
                if order_index == 0
                else int(rng.integers(days_since_last_order, history_days + 1))
            )
            order_date = observation_at - timedelta(days=elapsed)
            amount = round(float(rng.uniform(12.0, 250.0)), 2)
            historical_amounts.append(amount)
            collections["orders"].append(
                {
                    "_id": f"{customer_id}:history:{order_index}",
                    "customer_id": customer_id,
                    "occurred_at": order_date,
                    "status": "completed",
                    "amount": amount,
                    "synthetic": True,
                }
            )
        features = {
            "age_years": age_years,
            "orders_365d": orders_365d,
            "days_since_last_order": days_since_last_order,
            "avg_order_value": (
                round(float(np.mean(historical_amounts)), 2)
                if historical_amounts
                else None
            ),
            "support_tickets_90d": support_tickets,
            "plan_tier": plan,
        }
        collections["customer_features"].append(
            {
                "_id": snapshot_id,
                "snapshot_id": snapshot_id,
                "customer_id": customer_id,
                "observation_at": observation_at,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "synthetic": True,
                "features": features,
            }
        )

        purchase_logit = (
            -1.15
            + 0.17 * orders_365d
            - 0.004 * days_since_last_order
            - 0.12 * support_tickets
            + 0.15 * (plan == "premium")
        )
        purchase_probability = 1.0 / (1.0 + np.exp(-purchase_logit))
        purchase = int(rng.random() < purchase_probability)
        churn_logit = (
            -2.0
            + 0.006 * days_since_last_order
            + 0.32 * support_tickets
            - 0.15 * orders_365d
            - 0.1 * (plan == "premium")
        )
        churn_probability = 1.0 / (1.0 + np.exp(-churn_logit))
        churn = int(rng.random() < churn_probability)

        outcome_start = observation_at
        outcome_end = observation_at + timedelta(days=OUTCOME_DAYS)
        if purchase:
            collections["orders"].append(
                {
                    "_id": f"{customer_id}:outcome",
                    "customer_id": customer_id,
                    "occurred_at": outcome_start
                    + timedelta(days=int(rng.integers(0, OUTCOME_DAYS))),
                    "status": "completed",
                    "amount": round(float(rng.uniform(12.0, 250.0)), 2),
                    "synthetic": True,
                }
            )
        if churn:
            collections["churn_events"].append(
                {
                    "_id": f"{customer_id}:synthetic-churn",
                    "customer_id": customer_id,
                    "occurred_at": outcome_start
                    + timedelta(days=int(rng.integers(0, OUTCOME_DAYS))),
                    "event_type": "synthetic_churn",
                    "synthetic": True,
                }
            )

        for target, label in (("purchase", purchase), ("churn", churn)):
            collections["training_labels"].append(
                {
                    "_id": f"{snapshot_id}:{target}",
                    "snapshot_id": snapshot_id,
                    "customer_id": customer_id,
                    "target": target,
                    "label": label,
                    "observation_at": observation_at,
                    "outcome_start": outcome_start,
                    "outcome_end": outcome_end,
                    "available_at": outcome_end
                    + timedelta(days=LABEL_DELAY_DAYS),
                    "synthetic": True,
                }
            )

    return collections


def load_synthetic_data(database, count: int = 720, seed: int = SEED) -> dict:
    collections = make_synthetic_documents(count=count, seed=seed)
    for name, documents in collections.items():
        if documents:
            database[name].bulk_write(
                [
                    ReplaceOne({"_id": document["_id"]}, document, upsert=True)
                    for document in documents
                ],
                ordered=False,
            )
    return {
        name: len(documents)
        for name, documents in collections.items()
    }


def rows_for_target(database, target: str) -> list[dict]:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    pipeline = [
        {"$match": {"target": target, "synthetic": True}},
        {
            "$lookup": {
                "from": "customer_features",
                "localField": "snapshot_id",
                "foreignField": "snapshot_id",
                "as": "snapshot",
            }
        },
        {"$unwind": "$snapshot"},
        {
            "$project": {
                "_id": 0,
                "snapshot_id": 1,
                "label": 1,
                "observation_at": 1,
                "outcome_start": 1,
                "outcome_end": 1,
                "available_at": 1,
                "features": "$snapshot.features",
                "feature_schema_version": "$snapshot.feature_schema_version",
            }
        },
    ]
    rows = list(database.training_labels.aggregate(pipeline))
    for row in rows:
        row["features"] = {name: row["features"][name] for name in FEATURE_COLUMNS}
    return rows
