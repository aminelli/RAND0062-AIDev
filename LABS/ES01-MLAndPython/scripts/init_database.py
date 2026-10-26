from customer_ml.database import get_database, initialize_database


def main():
    database = get_database()
    database.command("ping")
    initialize_database(database)
    print("Database reachable; collections and indexes initialized.")


if __name__ == "__main__":
    main()
