import os
from urllib.parse import unquote, urlparse

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

from customer_ml.api import create_app
from customer_ml.config import TARGETS
from customer_ml.database import COLLECTIONS, initialize_database
from customer_ml.synthetic import load_synthetic_data
from customer_ml.training import train_target


@pytest.mark.integration
def test_seed_train_infer_and_persist(tmp_path):
    uri = os.getenv("MONGO_TEST_URI")
    if not uri:
        pytest.skip("Set MONGO_TEST_URI to an isolated database ending in _test")
    database_name = unquote(urlparse(uri).path.strip("/").split("/")[0])
    if not database_name.endswith("_test"):
        pytest.fail("MONGO_TEST_URI must name a dedicated database ending in _test")

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    database = client[database_name]
    try:
        database.command("ping")
        for collection in COLLECTIONS:
            database[collection].drop()
        initialize_database(database)
        load_synthetic_data(database, count=720, seed=20261025)

        for target in TARGETS:
            train_target(database, target, tmp_path, seed=20261025)

        api = TestClient(create_app(lambda: database, tmp_path))
        response = api.post("/predictions/purchase/synthetic-000000")

        assert response.status_code == 200
        result = response.json()
        assert result["synthetic"] is True
        assert result["target"] == "purchase"
        assert database.predictions.count_documents(
            {"customer_id": "synthetic-000000", "target": "purchase"}
        ) == 1
    finally:
        for collection in COLLECTIONS:
            database[collection].drop()
        client.close()
