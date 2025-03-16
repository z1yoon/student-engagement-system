import os
import pyodbc

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

def drop_all_tables():
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
        """)

        tables = [row[0] for row in cursor.fetchall()]


        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE {table};")
            print(f"✅ Dropped table: {table}")

        conn.commit()
        conn.close()
        print("✅ All tables dropped successfully.")

    except Exception as e:
        print(f"❌ Error dropping tables: {e}")

drop_all_tables()

