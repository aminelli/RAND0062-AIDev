import os

from customer_ml.config import MODEL_DIR, TARGETS
from customer_ml.database import get_database
from customer_ml.training import train_target


def main():
    database = get_database()
    database.command("ping")
    seed = int(os.getenv("DEMO_SEED", "20261025"))
    for target in TARGETS:
        metadata = train_target(database, target, MODEL_DIR, seed=seed)
        print(
            f"{target}: model={metadata['model_version']}, "
            f"test metrics={metadata['metrics']}"
        )


if __name__ == "__main__":
    main()
