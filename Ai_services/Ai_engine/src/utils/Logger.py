"""
File : logger.py
Description : Class is responsible for logging info and errors
Created on : 17-Feb-2025
Author : HCLTech
Includes:
- Daily rotating file handlers with custom retention.
- Console logging for real-time log streaming.
- Custom filters for log level segregation.
- Dynamic log file naming based on retention days.
"""

import os
import time
import logging
from src.constant.constants import Constants
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo  

debug_log_flag = os.getenv(Constants.ENABLE_LOG_DEBUG, Constants.TRUE).lower() == Constants.TRUE
error_log_flag = os.getenv(Constants.ENABLE_LOG_ERROR, Constants.TRUE).lower() == Constants.TRUE
critical_log_flag = os.getenv(Constants.ENABLE_LOG_CRITICAL, Constants.TRUE).lower() == Constants.TRUE
info_log_level = os.getenv(Constants.ENABLE_LOG_INFO, Constants.TRUE).lower() == Constants.TRUE
warning_log_level = os.getenv(Constants.ENABLE_LOG_WARNING, Constants.TRUE).lower() == Constants.TRUE
retention_days = int(os.getenv(Constants.LOG_RETENTION_DAYS, str(Constants.ONE)))


# Timezone for logging
mexico_zone = ZoneInfo(Constants.TIME_ZONE_INFO)
import logging
from datetime import datetime, timezone

class MexicoFormatter(logging.Formatter):
    LOCATION_WIDTH = 42 

    def converter(self, timestamp):
        return datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).astimezone(mexico_zone)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        location = f"{record.filename}.{record.funcName}"

        if len(location) > self.LOCATION_WIDTH:
            location = location[: self.LOCATION_WIDTH - 3] + "..."

        record.location = location.ljust(self.LOCATION_WIDTH)

        return super().format(record)


class LoggingConfig:

    def __init__(self):
        self.logger = logging.getLogger()

    def setup_logging(self):
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(logging.DEBUG)

        log_directory = Constants.LOGGER_ROOT_FOLDER_NAME
        os.makedirs(log_directory, exist_ok=True)

        date_range = self.get_log_date_range()
        debug_log_file = os.path.join(log_directory, f'debug_{date_range}.log')
        info_log_file = os.path.join(log_directory, f'info_{date_range}.log')

        debug_handler = TimedRotatingFileHandler(debug_log_file, when=Constants.MIDNIGHT, interval=Constants.ONE, backupCount=retention_days)
        info_handler = TimedRotatingFileHandler(info_log_file, when=Constants.MIDNIGHT, interval=Constants.ONE, backupCount=retention_days)
        console_handler = logging.StreamHandler()

        debug_handler.setLevel(logging.DEBUG)
        info_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        time_format = '%Y-%m-%d %H:%M:%S'

        debug_formatter = MexicoFormatter("%(asctime)s | %(filename)s.%(funcName)-25s <-> [%(levelname)-5s] %(message)s", datefmt=time_format)
        info_formatter = MexicoFormatter("%(asctime)s | %(filename)s.%(funcName)-25s | [%(levelname)-5s] | %(message)s", datefmt=time_format)
        console_formater = MexicoFormatter("%(asctime)s [%(levelname)-5s]  %(message)s")
        debug_handler.setFormatter(debug_formatter)
        info_handler.setFormatter(info_formatter)
        console_handler.setFormatter(console_formater)

        debug_handler.addFilter(self.DebugFilter())
        info_handler.addFilter(self.InfoFilter())
        console_handler.addFilter(self.InfoFilter())

        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)

        self.suppress_third_party_logs()
        return self.logger

    def get_log_date_range(self):
        """Return a string for log filename based on retention_days."""
        today = datetime.now(mexico_zone)
        if retention_days <= 1:
            return today.strftime('%Y-%m-%d')
        else:
            end_date = today + timedelta(days=retention_days - 1)
            return f"{today.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"

    def suppress_third_party_logs(self):
        noisy_loggers = [
            "pika", "aio_pika", "asyncio",
            "pika.adapters.utils.connection_workflow", "_AsyncTransportBase",
            "pika.adapters","pika.adapters.blocking_connection","pika.adapters.utils","pika.adapters.utils.selector_ioloop_adapter",
        ]

        for name in noisy_loggers:
            log = logging.getLogger(name)
            log.setLevel(logging.CRITICAL)  
            log.propagate = False

    class DebugFilter(logging.Filter):
        def filter(self, record):
            if debug_log_flag:
                return record.levelno in [logging.DEBUG, logging.ERROR, logging.CRITICAL]
            return record.levelno in [logging.ERROR, logging.CRITICAL]

    class InfoFilter(logging.Filter):
        def filter(self, record):
            if info_log_level:
                return record.levelno == logging.INFO
            elif warning_log_level:
                return record.levelno == logging.WARNING
            return record.levelno in [logging.INFO, logging.WARNING]

def log_time(msg=None,info_log_level = False):
    """
    Decorator that logs the time taken by a function with a custom message.

    Args:
        msg (str): Custom message to include in logs.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            if info_log_level:
                logging.info(f"[{func.__name__}:-{msg}] executed in {elapsed:.4f} seconds")
            else:
                logging.debug(f"[{func.__name__}:-{msg}] executed in {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator