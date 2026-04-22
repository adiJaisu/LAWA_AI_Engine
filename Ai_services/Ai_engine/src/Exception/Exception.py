class BaseException(Exception):
    def __init__(self, message="An error occurred", status_code=500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

# General Exceptions
class FileNotFoundException(BaseException):
    def __init__(self, message="File not found", status_code=404):
        super().__init__(message, status_code)

class InvalidInputException(BaseException):
    def __init__(self, message="Invalid input provided", status_code=400):
        super().__init__(message, status_code)

class ModelLoadingError(BaseException):
    def __init__(self, message="Error loading model", status_code=500):
        super().__init__(message, status_code)

class PredictionError(BaseException):
    def __init__(self, message="Prediction error", status_code=500):
        super().__init__(message, status_code)

class RTSPConnectionError(BaseException):
    def __init__(self, message="RTSP connection error", status_code=503):
        super().__init__(message, status_code)

class AnnotationError(BaseException):
    def __init__(self, message="Error in annotation processing", status_code=500):
        super().__init__(message, status_code)

# Video Processing Exceptions
class VideoProcessingError(BaseException):
    def __init__(self, message="Video processing error", status_code=500):
        super().__init__(message, status_code)

class FrameProcessingError(VideoProcessingError):
    def __init__(self, message="Error processing frame", status_code=500):
        super().__init__(message, status_code)

class FrameProcessingException(BaseException):
    def __init__(self, message="Error processing video frames", status_code=500):
        super().__init__(message, status_code)

class MotionDetectionException(BaseException):
    def __init__(self, message="Motion detection error", status_code=500):
        super().__init__(message, status_code)

# Specific Detection Exceptions
class DistanceCalculationError(BaseException):
    def __init__(self, message="Distance calculation error", status_code=500):
        super().__init__(message, status_code)

class EmptyQueueException(BaseException):
    def __init__(self, message="Queue is empty", status_code=422):
        super().__init__(message, status_code)

class DetectorInitializationError(BaseException):
    def __init__(self, message="Detector initialization error", status_code=500):
        super().__init__(message, status_code)

class RabbitMQConnectionError(BaseException):
    def __init__(self, message="RabbitMQ connection error", status_code=503):
        super().__init__(message, status_code)

class PersonDetectionError(BaseException):
    def __init__(self, message="person camera detection error", status_code=500):
        super().__init__(message, status_code)

class InvalidDetectionError(BaseException):
    def __init__(self, message="Invalid detection error", status_code=422):
        super().__init__(message, status_code)

class ConfigurationError(BaseException):
    def __init__(self, message="Configuration error", status_code=500):
        super().__init__(message, status_code)

class InvalidInputError(BaseException):
    def __init__(self, message="Invalid input data", status_code=400):
        super().__init__(message, status_code)