import os
import time
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
logger = LoggingConfig().setup_logging()

# Config
RETENTION_DAYS = int(os.getenv(Constants.LOG_RETENTION_DAYS, str(Constants.TWO)))
TIMEZONE = ZoneInfo(Constants.TIME_ZONE_INFO)


def delete_old_logs():
    now = datetime.now(TIMEZONE)
    cutoff = now - timedelta(days=RETENTION_DAYS)

    logger.info(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Cleaning logs older than {cutoff.strftime('%Y-%m-%d %H:%M:%S')}")

    for root, _, files in os.walk(Constants.LOGGER_ROOT_FOLDER_NAME):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=TIMEZONE)
                if mtime < cutoff:
                    os.remove(file_path)
                    logger.info(f"[LogCleanup] Deleted {file_path}")
            except Exception as e:
                logger.info(f"[LogCleanup] Error deleting {file_path}: {e}")

def run_cleanup_loop():
    has_run_today = False

    while True:
        now = datetime.now(TIMEZONE)
        hour, minute = now.hour, now.minute

        if hour == Constants.TARGET_HOUR and minute == Constants.TARGET_MINUTE:
            if not has_run_today:
                delete_old_logs()
                has_run_today = True
        else:
            has_run_today = False

        time.sleep(30)

def start_log_cleanup_thread():
    thread = threading.Thread(target=run_cleanup_loop, daemon=True)
    thread.start()