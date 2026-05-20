import psycopg2
import json
import time
from src.utils.logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.exception.exception import DatabaseConnectionError

logger = LoggingConfig.setup_logging()

class PostgresManager:
    """
    Handles connections to the PostgreSQL database for the AI VMS system.
    """
    def __init__(self):
        self.host = cfg.get_env_config(Constants.DB_HOST)
        self.port = cfg.get_env_config(Constants.DB_PORT)
        self.database = cfg.get_env_config(Constants.DB_NAME)
        self.user = cfg.get_env_config(Constants.DB_USER)
        self.password = cfg.get_env_config(Constants.DB_PASSWORD)
        self.initialize_database()

    def initialize_database(self):
        """
        Creates the required tables if they don't exist.
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                create_table_query = """
                    CREATE TABLE IF NOT EXISTS events (
                        id SERIAL PRIMARY KEY,
                        camera_id VARCHAR(255),
                        usecase_name VARCHAR(255),
                        evidence_path TEXT,
                        event_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("[PostgresManager] Database tables initialized successfully.")
        except Exception as e:
            logger.error(f"[PostgresManager] Failed to initialize database: {e}")
        finally:
            if conn:
                conn.close()

    def get_connection(self, max_retries=5, delay=2):
        """
        Retrieves a connection to PostgreSQL with retry logic.
        """
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    dbname=self.database,
                    user=self.user,
                    password=self.password,
                    connect_timeout=5
                )
                return conn
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[PostgresManager] Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"[PostgresManager] Error connecting to PostgreSQL after {max_retries} attempts: {e}", exc_info=True)
                    raise DatabaseConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def insert_event(self, camera_id, usecase_name, evidence_path, event_data):
        """
        Inserts a new event into the `events` table.
        This explicitly avoids checking for previous events and just inserts the batch.
        """
        try:
            conn = self.get_connection()
        except DatabaseConnectionError:
            logger.error(f"[PostgresManager] Failed to insert event for {usecase_name} from {camera_id} due to connection failure.")
            raise

        try:
            with conn.cursor() as cursor:
                insert_query = """
                    INSERT INTO events (camera_id, usecase_name, evidence_path, event_data)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """
                # psycopg2 handles dicts correctly with Json adapter or we can pass as json.dumps string
                cursor.execute(insert_query, (
                    camera_id, 
                    usecase_name, 
                    evidence_path, 
                    json.dumps(event_data)
                ))
                
                event_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"[PostgresManager] Successfully inserted event with ID: {event_id} for camera: {camera_id}")
                return event_id
        except Exception as e:
            logger.error(f"[PostgresManager] Exception during event insertion: {e}", exc_info=True)
            conn.rollback()
            raise DatabaseConnectionError(f"Exception during event insertion: {e}")
        finally:
            if conn:
                conn.close()
