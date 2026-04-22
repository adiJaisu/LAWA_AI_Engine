import os
import numpy as np
import cv2

class Constants:
    # Directory and File Paths
    # PARENT_DIR = os.path.abspath(os.curdir)
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CONFIG_FILE_PATH = os.path.join(PARENT_DIR, "src/config/config.ini")
    
    # gpu variable constants
    WORKER = "worker"
    GPU = "gpu"
    GPU_ID = "gpu_id"
    THREAD = "thread"
    WORKER_ID = "worker_id"
    GPU_MANAGER = "gpu_manager"
    MODEL = "model"
    SECONDARY_MODEL = "secondary_model"
    BUSY = "BUSY"
    IDLE = "IDLE"
    SHUTDOWN = "SHUTDOWN"
    NUMBERS_OF_WORKER = "NUMBERS_OF_WORKER"
    NUMBERS_OF_WORKER_LESS_GPU = "NUMBERS_OF_WORKER_LESS_GPU"
    NUM_CONSUMER_THREADS = "NUM_CONSUMER_THREADS"
    MAX_QUEUE_SIZE = "MAX_QUEUE_SIZE"
    DECODE_WORKERS = "DECODE_WORKERS"
    
    # Logging Configuration
    MAX_BYTES = 10000000
    BACKUP_COUNT = 99
    ALL_PERMISSION = 0o777
    LOG_FILE_NAME = 'run_log-{}.log'
    LOGGER_ROOT_FOLDER_NAME = 'logs'
    ROOT_DIR_PATH = os.path.abspath(os.curdir)
    UTF_8_ENCODING = "UTF-8"
    SUCCESS = "SUCCESS"
    ENABLE_LOG_DEBUG = "ENABLE_LOG_DEBUG"
    ENABLE_LOG_ERROR = "ENABLE_LOG_ERROR"
    MIDNIGHT = "midnight"
    ENABLE_LOG_CRITICAL = "ENABLE_LOG_CRITICAL"
    ENABLE_LOG_INFO = "ENABLE_LOG_INFO"
    ENABLE_LOG_WARNING = "ENABLE_LOG_WARNING"
    LOG_RETENTION_DAYS = "LOG_RETENTION_DAYS"
    LOG_LEVEL = "LOG_LEVEL"
    TRUE= "true"
    FALSE="false"
    TIME_ZONE_INFO = "America/Mexico_City"
    TIME_ZONE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
    UNDER_SCORE = '_'
    TARGET_HOUR = 0
    TARGET_MINUTE = 0
    DEBUG = "debug"
    INFO = 'info'
    
    PERSON = "person"
    BOXES = "boxes"
    CONF = "conf"
    
    BBOX = 'bbox'
    CLASS = 'class'
    DISTANCE = 'distance'
    VIDEO = 'video'
    CONFIDENCE = "confidence"
    CLASS_ID = "class_id"
    
    # Numeric Constants
    ZERO_POINT_THREE = 0.3
    ZERO_POINT_ONE = 0.1
    ZERO_POINT_FOUR = 0.4
    ZERO_POINT_FIVE = 0.5
    ZERO_POINT_SIX = 0.6
    ZERO_POINT_NINE = 0.9
    ZERO_POINT_NINE_EIGHT = 0.98
    TWO_POINT_FIVE = 2.5
    ZERO_POINT_TWO_FIVE = 0.25
    ONE_FLOAT = 1.0
    NINE = 9
    TEN = 10
    MINUS_TEN = -10
    FIFTEEN = 15
    TWENTY = 20
    TWENTY_FIVE = 25
    THIRTY = 30
    FORTY = 40
    FIFTY = 50
    ONE_HUNDRED = 100
    ONE_HUNDRED_FLOAT = 100.0
    TWO_HUNDRED_FIFTY = 250
    THREE_HUNDRED = 300
    TWO_FIVE_FIVE = 255
    ONE_THOUSAND_TWENTY_FOUR = 1024
    MINUS_ONE = -1
    MINUS_TWO = -2
    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT , NINE, TEN ,ELEVEN = range(12)
    ZERO_FLOAT = 0.0
    ZERO_POINT_FORTY_FIVE = 0.45
    TEN_MS = 0.01
    ONE_SEC = 1
    FIVE_SEC = 5
    THREE_SEC = 3
    TEN_SEC = 10
    
    # Video Parameters
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 5
    FRAME_SIZE = "frame_size"
    FRAME_SIZE_H = "frame_height"
    FRAME_SIZE_W = "frame_width"
    FRAME="frame"
    MP4_CODEC='mp4v'
    
    # config environments
    LOITERING_DETECTION = "LOITERING_DETECTION"
    TAILGATE_DETECTION = "TAILGATE_DETECTION"
    CROWD_DENSITY = "CROWD_DENSITY"      #add
    CROWD_LIMIT = "CROWD_LIMIT"           # add
    IN_OUT_PERSON_COUNT = "IN_OUT_PERSON_COUNT"
    DEFAULT_ENVIRONMENT="DEFAULT"
    DEVICE = "DEVICE"
    AVAILABLE_GPU_INDEX = "AVAILABLE_GPU_INDEX"
    PT_MODEL_TYPE = '.pt'
    TORCH_MODEL_TYPE = 'rfdter'
    SEG_MODEL = 'seg'
    
    # use case constant
    LOITERING_DETECTION_USECASE = "Loitering_Detection"
    TAILGATE_DETECTION_USECASE = "Tailgate_Detection"
    IN_OUT_PERSON_COUNT_USECASE = "In_Out_Person_Count"
    CROWD_DENSITY_USECASE = "Crowd_Density"               # add
    
    LOITERING_DETECTION_MODEL_PATH = "LOITERING_DETECTION_MODEL_PATH"
    TAILGATE_DETECTION_MODEL_PATH = "TAILGATE_DETECTION_MODEL_PATH"
    IN_OUT_PERSON_COUNT_MODEL_PATH = "IN_OUT_PERSON_COUNT_MODEL_PATH"
    CROWD_DENSITY_MODEL_PATH = "CROWD_DENSITY_MODEL_PATH"           # add
    PERSON_CLASS_ID="PERSON_CLASS_ID"

    CASE = "CASE"
    PIXEL_TO_METER = "PIXEL_TO_METER"
    MAX_ALLOWED_MOVEMENT = "MAX_ALLOWED_MOVEMENT"
    GRACE_PERIOD_NRUA = "GRACE_PERIOD_NRUA"
    GRACE_PERIOD_NRPA = "GRACE_PERIOD_NRPA"
    GRACE_PERIOD_RPA = "GRACE_PERIOD_RPA"
    TRACKER_NAME = "TRACKER_NAME"
    LOITERING_DETECTION_CONFIDENCE = "LOITERING_DETECTION_CONFIDENCE"
    TAILGATE_DETECTION_CONFIDENCE = "TAILGATE_DETECTION_CONFIDENCE"
    IN_OUT_CONFIDENCE = "IN_OUT_CONFIDENCE"
    CROWD_DENSITY_CONFIDENCE = "CROWD_DENSITY_CONFIDENCE"               # add
    TAILGATE_MIN_TAILGATE_COUNT = "TAILGATE_MIN_TAILGATE_COUNT"
    TAILGATE_MIN_CONSECUTIVE_FRAMES = "TAILGATE_MIN_CONSECUTIVE_FRAMES"
    TAILGATE_MIN_SUSTAIN_FRAMES = "TAILGATE_MIN_SUSTAIN_FRAMES"
    TAILGATE_MIN_PUNCH_GAP = "TAILGATE_MIN_PUNCH_GAP"
    TAILGATE_ROI_MIN_FRAMES = "TAILGATE_ROI_MIN_FRAMES"
    TAILGATE_GREEN_MIN_FRAMES = "TAILGATE_GREEN_MIN_FRAMES"
    TAILGATE_GLOBAL_ODC = "TAILGATE_GLOBAL_ODC"
    
    CUDA="cuda"
    OPENCV = "opencv"
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    X, Y, W, H = 'x', 'y', 'w', 'h'
    
    # RabbitMQ Server Configuration
    RABBITMQ_HOST = 'RABBITMQ_HOST'
    RABBITMQ_SERVICE = 'rabbitmq_service'
    RABBITMQ_PORT = 'RABBITMQ_PORT'
    RABBITMQ_USERNAME = 'RABBITMQ_USERNAME'
    RABBITMQ_PASSWORD = 'RABBITMQ_PASSWORD'
    QUEUE_NAME = 'QUEUE_NAME'
    EXCHANGE_NAME = 'EXCHANGE_NAME'
    ROUTING_KEY = 'ROUTING_KEY'
    EXCHANGE_TYPE = 'EXCHANGE_TYPE'
    DURABLE = 'DURABLE'
    FRAME_SHAPE = "frame_shape"
    USECASE = "usecase"
    MAX_MESSAGES = "MAX_MESSAGES"
    USERNAME = "username"
    PASSWORD = "password"
    MAX_MSG_SIZE= 2000
    UTF_8= "utf-8"
    X_MAX_LENGTH = "x-max-length"
    CONNECTION_NAME = "connection_name"
    COMPONENT_NAME = "component"
    CONNECTION_HEARTBEAT = 600
    BLOCKED_CONNECTION_TIMEOUT = 300
    MAX_CONNECTION_RETRY_DURATION = 3600        
    CONNECTION_RETRY_DELAY = 5                

    COMPONENTS_MAPPING = {
        "queue_loitering_detection":"ai-loitering-detection",
        "queue_tailgate_detection":"tailgate",
        "queue_in_out_person_count":"in_out_person_count",
        "queue_crowd_density":"crowd_density"                          # add
    }
 
    # Video and Camera Metadata
    VIDEO_WRITER = "video_writer"
    VIDEO_PATH = "video_path"
    CAMERA_METADATA = "camera_metadata"
    CAMERA_ID = "camera_id"
    CAMERA_HEIGHT = "height"
    RTSP_URL = "rtsp_url"
    CAMERA_NAME = "name"
    CODEC = "codec"
    ROIS = "rois"
    ROIS_LIST= "roiList"
    RESOLUTION = "resolution"
    LOCATION_ID = "location_id"
    NOTES = "notes"

    # detection constants
    DETECTION_DETAILS = "detection_details"

    # frame quality constants
    TOTAL_CELLS = 'total_cells'
    BAD_CELLS = 'bad_cells'
    BAD_AREA_PERCENTAGE = 'bad_area_percentage'
    IS_BAD_FRAME = 'is_bad_frame'
    BOUNDS = 'bounds'
    IS_BAD = 'is_bad'
    METRICS = 'metrics'
    ISSUES = 'issues'
 
    # Frame Metadata
    FRAME_METADATA = "frame_metadata"
    FRAME_ID = "frame_id"
    TIME_STAMP = "time_stamp"
    USECASE_TIMESTAMP = "usecase_time_stamp"
    ALERT = "ALERT"
    FRAME_QUALITY = "frame_quality"
    DETECTIONS="detection"
    DETECTED_TIME = "detected_time"
    USECASE_NAME = "usecase_name"
    TO_SKIP = "to_skip"
    METADATA = "metadata"
    RABBITMQ_SENT_TIMING="rabbitmq_sent_timing"
    RABBITMQ_CONSUME_TIMING="rabbitmq_consume_timing"
    RAW_FRAME = 'raw_frame'
    
    # Global Configuration Keys
    PARAMETER = "parameter"
    VALUE = "value"
    CONFIG_METADATA = "config_metadata"
    PERSON_DETECTION_CONFIDENCE="PERSON DETECTION CONFIDENCE"
    
    FRAME_SENT_QUALITY_TO_EVENT_MANAGER = 'FRAME_SENT_QUALITY_TO_EVENT_MANAGER'
    FRAME_QUALITY_THRESHOLDS = 'FRAME_QUALITY_THRESHOLDS'
    BLUR_THRESHOLD = 'BLUR_THRESHOLD'
    MIN_CONTRAST = 'MIN_CONTRAST'
    VARIANCE_THRESHOLD = 'VARIANCE_THRESHOLD'
    GRADIENT_THRESHOLD = 'GRADIENT_THRESHOLD'
    GRID_SIZE = 'GRID_SIZE'
    BLURRY = 'blurry'
    LOW_CONTRAST = 'low_contrast'
    LOW_VARIANCE = 'low_variance'
    LOW_GRADIENT = 'low_gradient'
    BLUR_VALUE = 'blur_value'
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    VARIANCE = 'variance'
    AVG_GRADIENT = 'avg_gradient'

    EVENT_MANAGER_QUEUE="EVENT_MANAGER_QUEUE"
    
    # Queue Constants
    LOITERING_DETECTION_QUEUE = "queue_loitering_detection"
    TAILGATE_DETECTION_QUEUE = "queue_tailgate_detection"
    IN_OUT_PERSON_COUNT_QUEUE = "queue_in_out_person_count"
    CROWD_DENSITY_QUEUE = "queue_crowd_density"             # add
 
    # Usecase to Queue Mapping
    USECASE_QUEUE_MAPPING = {
        LOITERING_DETECTION_QUEUE:LOITERING_DETECTION_USECASE,
        TAILGATE_DETECTION_QUEUE:TAILGATE_DETECTION_USECASE,
        IN_OUT_PERSON_COUNT_QUEUE:IN_OUT_PERSON_COUNT_USECASE,
        CROWD_DENSITY_QUEUE:CROWD_DENSITY_USECASE    # add
    }

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
