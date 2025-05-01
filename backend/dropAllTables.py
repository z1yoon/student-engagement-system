import os
import pyodbc

def drop_all_tables():
    DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
    
    if not DB_CONNECTION_STRING:
        print("❌ DB_CONNECTION_STRING environment variable not set.")
        return
    
    try:
        # Connect to the database
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        
        # First get a list of all foreign key constraints
        print("Finding and dropping all foreign key constraints...")
        cursor.execute("""
            SELECT 
                'ALTER TABLE ' + QUOTENAME(OBJECT_SCHEMA_NAME(f.parent_object_id)) + '.' +
                QUOTENAME(OBJECT_NAME(f.parent_object_id)) + ' DROP CONSTRAINT ' + QUOTENAME(f.name) AS drop_script
            FROM sys.foreign_keys AS f
        """)
        
        # Drop all foreign key constraints first
        for row in cursor.fetchall():
            drop_script = row[0]
            try:
                print(f"Executing: {drop_script}")
                cursor.execute(drop_script)
            except Exception as e:
                print(f"❌ Error dropping constraint: {e}")
        
        # Now get all tables to drop
        cursor.execute("""
            SELECT 
                QUOTENAME(TABLE_SCHEMA) + '.' + QUOTENAME(TABLE_NAME) AS table_name
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' 
              AND TABLE_NAME != 'sysdiagrams'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Drop each table
        for table in tables:
            try:
                print(f"Dropping table {table}...")
                cursor.execute(f"DROP TABLE {table}")
            except Exception as e:
                print(f"❌ Error dropping table {table}: {e}")
        
        conn.commit()
        print("✅ Tables dropped successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    drop_all_tables()

