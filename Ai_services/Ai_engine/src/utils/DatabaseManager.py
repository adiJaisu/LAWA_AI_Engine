import sqlite3
import os
from datetime import datetime
from src.constant.constants import Constants
from src.utils.Logger import LoggingConfig

logger = LoggingConfig().setup_logging()

class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Place database in the project root
            db_path = os.path.join(Constants.PARENT_DIR, "lawa_ai.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database schema."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Employee Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS employees (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT UNIQUE,
                        card_id TEXT,
                        name TEXT,
                        odc_no TEXT
                    )
                ''')
                # Access Logs Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS access_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT,
                        name TEXT,
                        odc_no TEXT,
                        status TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                logger.debug(f"Database initialized/verified at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def get_employee_by_count(self, count):
        """Fetches employee by sequential ID (matching original prototype logic)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT employee_id, card_id, name, odc_no FROM employees WHERE id = ?", (count,))
                row = cursor.fetchone()
                if row:
                    return {
                        "employee_id": row[0],
                        "card_id": row[1],
                        "name": row[2],
                        "odc_no": row[3]
                    }
                return None
        except Exception as e:
            logger.error(f"Error fetching employee: {e}")
            return None

    def log_access(self, employee_data, status="SUCCESS"):
        """Logs an access attempt."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO access_logs (employee_id, name, odc_no, status)
                    VALUES (?, ?, ?, ?)
                ''', (
                    employee_data.get("employee_id"),
                    employee_data.get("name"),
                    employee_data.get("odc_no"),
                    status
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging access: {e}")

    def add_employee(self, emp_id, card_id, name, odc_no):
        """Helper to add employees during migration."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO employees (employee_id, card_id, name, odc_no)
                    VALUES (?, ?, ?, ?)
                ''', (emp_id, card_id, name, odc_no))
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding employee: {e}")
