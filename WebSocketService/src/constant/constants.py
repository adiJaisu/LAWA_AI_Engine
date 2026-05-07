import os

class Constants:
    # Directory and File Paths
    PARENT_DIR = os.path.abspath(os.curdir)
    CONFIG_FILE_PATH = os.path.join(PARENT_DIR, "src/config/config.ini")
    UTF_8_ENCODING = "UTF-8"
    DEFAULT_ENVIRONMENT = "DEFAULT"

    # WebSocket Configuration Keys
    WEBSOCKET_HOST = "WEBSOCKET_HOST"
    WEBSOCKET_PORT = "WEBSOCKET_PORT"
