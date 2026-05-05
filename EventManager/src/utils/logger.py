import logging
import os
import sys

class LoggingConfig:
    """
    Configures and returns a centralized logger for the application.
    Provides standard formatting for better debug and info tracing.
    """
    _logger = None

    @classmethod
    def setup_logging(cls, level=logging.INFO):
        if cls._logger is not None:
            return cls._logger

        logger = logging.getLogger("EventManager")
        logger.setLevel(level)

        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "event_manager.log")

        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(level)
        f_handler.setLevel(level)

        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
        )
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        # Add handlers to the logger
        if not logger.handlers:
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        cls._logger = logger
        return cls._logger
