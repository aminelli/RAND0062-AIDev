from datetime import timedelta

from customer_ml.synthetic import make_synthetic_documents


def test_generator_is_deterministic_and_marks_data_synthetic():
    first = make_synthetic_documents(count=120, seed=42)
    second = make_synthetic_documents(count=120, seed=42)

    assert first == second
    assert len(first["customers"]) == 120
    assert all(document["synthetic"] for docs in first.values() for document in docs)
    for target in ("purchase", "churn"):
        labels = [
            document["label"]
            for document in first["training_labels"]
            if document["target"] == target
        ]
        assert set(labels) == {0, 1}


def test_labels_follow_the_future_observation_window():
    docs = make_synthetic_documents(count=120, seed=42)
    for label in docs["training_labels"]:
        assert label["outcome_start"] == label["observation_at"]
        assert label["outcome_end"] > label["observation_at"]
        assert label["available_at"] > label["outcome_end"]


def test_features_and_labels_match_their_synthetic_events():
    docs = make_synthetic_documents(count=120, seed=42)
    for snapshot in docs["customer_features"]:
        customer_orders = [
            order for order in docs["orders"]
            if order["customer_id"] == snapshot["customer_id"]
        ]
        historical = [
            order for order in customer_orders
            if snapshot["observation_at"] - timedelta(days=365)
            <= order["occurred_at"] < snapshot["observation_at"]
        ]
        assert len(historical) == snapshot["features"]["orders_365d"]

        labels = [
            row for row in docs["training_labels"]
            if row["snapshot_id"] == snapshot["snapshot_id"]
        ]
        for label in labels:
            outcome_orders = [
                order for order in customer_orders
                if label["outcome_start"] <= order["occurred_at"] < label["outcome_end"]
            ]
            if label["target"] == "purchase":
                assert label["label"] == int(bool(outcome_orders))
            else:
                events = [
                    event for event in docs["churn_events"]
                    if event["customer_id"] == snapshot["customer_id"]
                    and label["outcome_start"] <= event["occurred_at"] < label["outcome_end"]
                ]
                assert label["label"] == int(bool(events))
