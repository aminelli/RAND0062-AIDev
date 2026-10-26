import os
from urllib.parse import quote_plus


DATABASE_NAME = os.getenv("MONGO_DATABASE", "customer_demo")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
FEATURE_SCHEMA_VERSION = "1.0"
TARGETS = ("purchase", "churn")
FEATURE_COLUMNS = (
    "age_years",
    "orders_365d",
    "days_since_last_order",
    "avg_order_value",
    "support_tickets_90d",
    "plan_tier",
)


def mongo_uri() -> str:
    configured_uri = os.getenv("MONGO_URI")
    if configured_uri:
        return configured_uri

    username = quote_plus(os.getenv("MONGO_APP_USERNAME", "customer_app"))
    password = quote_plus(os.getenv("MONGO_APP_PASSWORD", ""))
    host = os.getenv("MONGO_HOST", "localhost")
    return (
        f"mongodb://{username}:{password}@{host}:27017/"
        f"{DATABASE_NAME}?authSource={DATABASE_NAME}"
    )
