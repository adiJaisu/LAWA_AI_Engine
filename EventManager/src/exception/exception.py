from src.utils.logger import LoggingConfig

logger = LoggingConfig.setup_logging()

class EventManagerException(Exception):
    """
    This class is responsible for custom exception handling in the event manager module.
    It automatically logs the error upon instantiation.
    """

    message = ''

    def __init__(self, message):
        self.message = message
        logger.error(self.message)

    def __str__(self):
        return repr(self.message)

class DatabaseConnectionError(EventManagerException):
    def __init__(self, message):
        super(DatabaseConnectionError, self).__init__(message)

class VideoProcessingError(EventManagerException):
    def __init__(self, message):
        super(VideoProcessingError, self).__init__(message)

class WebSocketError(EventManagerException):
    def __init__(self, message):
        super(WebSocketError, self).__init__(message)

class MessageProcessingError(EventManagerException):
    def __init__(self, message):
        super(MessageProcessingError, self).__init__(message)
