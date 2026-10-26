import os

from customer_ml.database import get_database, initialize_database
from customer_ml.synthetic import SEED, load_synthetic_data


def main():
    count = int(os.getenv("DEMO_CUSTOMERS", "720"))
    seed = int(os.getenv("DEMO_SEED", str(SEED)))
    database = get_database()
    database.command("ping")
    initialize_database(database)
    counts = load_synthetic_data(database, count=count, seed=seed)
    print(f"Loaded deterministic synthetic dataset (seed={seed}): {counts}")


if __name__ == "__main__":
    main()
