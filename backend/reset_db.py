# reset_db.py
from db import reset_database
import os

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")


if __name__ == "__main__":
    reset_database()
    print("Database has been reset.")

