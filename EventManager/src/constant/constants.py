import os

class Constants:
    # Directory and File Paths
    PARENT_DIR = os.path.abspath(os.curdir)
    CONFIG_FILE_PATH = os.path.join(PARENT_DIR, "src/config/config.ini")
    UTF_8_ENCODING = "UTF-8"
    DEFAULT_ENVIRONMENT = "DEFAULT"

    # RabbitMQ Configuration Keys
    RABBITMQ_HOST = "RABBITMQ_HOST"
    RABBITMQ_PORT = "RABBITMQ_PORT"
    RABBITMQ_USERNAME = "RABBITMQ_USERNAME"
    RABBITMQ_PASSWORD = "RABBITMQ_PASSWORD"
    EXCHANGE_NAME = "EXCHANGE_NAME"
    EXCHANGE_TYPE = "EXCHANGE_TYPE"

    # Usecase Names (Synced with AI Engine)
    LOITERING_DETECTION_USECASE = "Loitering_Detection"
    TAILGATE_DETECTION_USECASE = "Tailgate_Detection"
    IN_OUT_PERSON_COUNT_USECASE = "In_Out_Person_Count"
    TRAIN_ARRIVAL_DEPART_MONITOR_USECASE = "Train_Arrival_Depart_Monitor"
    CROWD_DENSITY_USECASE = "Crowd_Density"
    PERSON_ENTERED_INSIDE_TRAIN_USECASE = "Person_Entered_Inside_Train"
    RESTROOM_PERSON_TRACKING_USECASE = "Restroom_Person_Tracking"
    PERSON_COUNT_INSIDE_COMPARTMENT_USECASE = "Person_Count_Inside_Compartment"
    QUEUE_MANAGEMENT_USECASE = "Queue_Management"
    BIRD_EYE_VIEW_USECASE = "Bird_Eye_View"
    
    # Event Queues (Derived from the usecases)
    LOITERING_DETECTION_EVENT_QUEUE = "loitering_detection_event_queue"
    TAILGATE_DETECTION_EVENT_QUEUE = "tailgate_detection_event_queue"
    IN_OUT_PERSON_COUNT_EVENT_QUEUE = "in_out_person_count_event_queue"
    TRAIN_ARRIVAL_DEPART_MONITOR_EVENT_QUEUE = "train_arrival_depart_monitor_event_queue"
    CROWD_DENSITY_EVENT_QUEUE = "crowd_density_event_queue"
    PERSON_ENTERED_INSIDE_TRAIN_EVENT_QUEUE = "person_entered_inside_train_event_queue"
    RESTROOM_PERSON_TRACKING_EVENT_QUEUE = "restroom_person_tracking_event_queue"
    PERSON_COUNT_INSIDE_COMPARTMENT_EVENT_QUEUE = "person_count_inside_compartment_event_queue"
    QUEUE_MANAGEMENT_EVENT_QUEUE = "queue_management_event_queue"
    BIRD_EYE_VIEW_EVENT_QUEUE = "bird_eye_view_event_queue"

    # Storage
    EVIDENCE_STORAGE_PATH = "EVIDENCE_STORAGE_PATH"
    EVIDENCE_VIDEO_FPS = "EVIDENCE_VIDEO_FPS"

    # Database Configuration Keys
    DB_HOST = "DB_HOST"
    DB_PORT = "DB_PORT"
    DB_NAME = "DB_NAME"
    DB_USER = "DB_USER"
    DB_PASSWORD = "DB_PASSWORD"

    # WebSocket Configuration Keys
    WEBSOCKET_HOST = "WEBSOCKET_HOST"
    WEBSOCKET_PORT = "WEBSOCKET_PORT"

    # Event Data Keys
    CAMERA_ID = "camera_id"
    CAMERA_ASSET_ID = "camera_asset_id"
    SENSOR_EVENT_TIME = "event_time"
    USECASE_ID = "usecase_id"
    SENSOR_EVENT_TYPE = "event_type"
    REQUEST_STATUS = "request_status"
    META_LOCATION_ID = "location_id"
    SENSOR_EVENT_DATA = "event_data"
    
    # Detection Dictionary Mapping
    DETECTION_DICT = {
        "Loitering_Detection": "Loitering Detected",
        "Tailgate_Detection": "Tailgating Detected",
        "In_Out_Person_Count": "Person Crossed Line",
        "Train_Arrival_Depart_Monitor": "Train Arrived/Departed",
        "Crowd_Density": "High Crowd Density",
        "Person_Entered_Inside_Train": "Person Entered Train",
        "Restroom_Person_Tracking": "Restroom Usage Monitored",
        "Person_Count_Inside_Compartment": "Compartment Occupancy Alert",
        "Queue_Management": "Queue Wait Time Exceeded",
        "Bird_Eye_View": "Bird Eye View Monitored"
    }
