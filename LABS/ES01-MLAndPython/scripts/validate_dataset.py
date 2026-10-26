from customer_ml.config import TARGETS
from customer_ml.database import get_database
from customer_ml.synthetic import rows_for_target
from customer_ml.training import _split_chronologically
import pandas as pd


def main():
    database = get_database()
    database.command("ping")
    for target in TARGETS:
        rows = rows_for_target(database, target)
        if not rows:
            raise SystemExit(f"No labels found for {target}; generate demo data first.")
        frame = pd.DataFrame(
            {
                "label": [int(row["label"]) for row in rows],
                "observation_at": [row["observation_at"] for row in rows],
            }
        )
        train, validation, test = _split_chronologically(frame)
        print(
            f"{target}: total={len(rows)}, positives="
            f"{sum(row['label'] for row in rows)}, chronological="
            f"{len(train)}/{len(validation)}/{len(test)}"
        )


if __name__ == "__main__":
    main()
