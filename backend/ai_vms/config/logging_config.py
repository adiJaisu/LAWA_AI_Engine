"""
Configures logging for the application with file and console handlers.

Includes:
- Weekly rotating file handlers for debug and info logs.
- Console logging for real-time log streaming.
- Custom filters for log level segregation.
- Dynamic log file naming based on the current week.

Author: HCLTech
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta
 
class LoggingConfig:

    def __init__(self):
        self.logger = logging.getLogger()

    def setup_logging(self):
        """
        This function initializes a logger with DEBUG level and configures three
        handlers: a weekly rotating file handler for debug level logs, a weekly
        rotating file handler for info level logs, and a console handler for
        streaming logs. The log files are named dynamically based on the current
        week's start and end dates. Custom formatters are applied to each handler 
        to format the log messages appropriately. Filters are added to ensure that
        only log messages of certain levels are passed to the respective handlers.
    
        Loggers:
            - Debug logs are handled by a TimedRotatingFileHandler, capturing DEBUG,
            ERROR, and CRITICAL levels.
            - Info logs are handled by a TimedRotatingFileHandler, capturing INFO
            and WARNING levels.
            - Console logs are handled by a StreamHandler, capturing INFO and
            WARNING levels.
        """
    
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to prevent duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Ensure "logs" directory exists
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)

        week_start, week_end = self.get_week_dates()

        debug_log_file = os.path.join(log_directory, f'debug_{week_start}_to_{week_end}.log')
        info_log_file = os.path.join(log_directory, f'logger_{week_start}_to_{week_end}.log')

        debug_handler = TimedRotatingFileHandler(debug_log_file, when='W0', interval=1, backupCount=4)
        info_handler = TimedRotatingFileHandler(info_log_file, when='W0', interval=1, backupCount=4)
        console_handler = logging.StreamHandler()

        debug_handler.setLevel(logging.DEBUG)
        info_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        debug_log_format = '%(asctime)s - %(module)s.%(funcName)s - %(levelname)s <-> %(message)s'
        info_log_format = '%(asctime)s - %(levelname)s <-> %(message)s'

        time_format = '%Y-%m-%d %H:%M:%S'

        debug_formatter = logging.Formatter(debug_log_format, datefmt=time_format)
        info_formatter = logging.Formatter(info_log_format, datefmt=time_format)

        debug_handler.setFormatter(debug_formatter)
        info_handler.setFormatter(info_formatter)
        console_handler.setFormatter(info_formatter)

        debug_handler.addFilter(self.DebugFilter())
        info_handler.addFilter(self.InfoFilter())
        console_handler.addFilter(self.InfoFilter())

        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)

        return self.logger

    def get_week_dates(self):
        """
        Return the start and end dates of the current week as strings in the format
        '%Y-%m-%d'.
        The week starts on Monday and ends on Sunday.
    
        Returns:
            tuple: (start_date, end_date)
        """
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return start_of_week.strftime('%Y-%m-%d'), end_of_week.strftime('%Y-%m-%d')
 
    class DebugFilter(logging.Filter):
        def filter(self, record):
            """
            Determine if a log record should be passed through to the debug log.
    
            Args:
                record (logging.LogRecord): The log record to be filtered.
            Returns:
                bool: True if the log record's level is DEBUG, ERROR, or CRITICAL,
                    indicating it should be passed through to the debug log; 
                    False otherwise.
            """
            return record.levelno in [logging.DEBUG, logging.ERROR, logging.CRITICAL]
    
    class InfoFilter(logging.Filter):
        def filter(self, record):
            """
            Determine if a log record should be passed through to the info log.
    
            Args:
                record (logging.LogRecord): The log record to be filtered.
            Returns:
                bool: True if the log record's level is INFO or WARNING,
                    indicating it should be passed through to the info log; 
                    False otherwise.
            """
            return record.levelno in [logging.INFO, logging.WARNING]
    
