
import sqlite3
import time
import os

db_path = "mlflow.db"

def test_db():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("SELECT 1;")
        conn.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

print(f"Testing {db_path}...")
for i in range(100):
    if not test_db():
        print(f"Failed at iteration {i}")
        break
else:
    print("Passed 100 iterations")
