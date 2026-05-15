import json
import threading
import time
import websocket
from src.utils.logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.exception.exception import WebSocketError

logger = LoggingConfig.setup_logging()

class WebSocketEventNotifier:
    """
    Class to handle real-time event notifications via WebSockets as a client.
    Connects to a centralized WebSocket hub for broadcasting.
    """
    ws = None
    hub_url = None

    @staticmethod
    def broadcast_event(payload):
        """
        Sends structured event data to the centralized WebSocket hub.
        """
        if not WebSocketEventNotifier.ws or not WebSocketEventNotifier.ws.sock or not WebSocketEventNotifier.ws.sock.connected:
            logger.warning("[WebSocket] Hub not connected. Broadcasting skipped.")
            return False

        try:
            message = json.dumps(payload)
            logger.info(f"[WebSocket] Sending event to Hub: {message[:200]}...")
            WebSocketEventNotifier.ws.send(message)
            logger.info("[WebSocket] Message successfully delivered to Hub.")
            return True
        except Exception as e:
            logger.error(f"[WebSocket] Failed to send message to Hub: {e}", exc_info=True)
            return False

    @staticmethod
    def initialize_websocket_client():
        """
        Starts the WebSocket client in a background thread with auto-reconnection.
        """
        def run_client():
            while True:
                try:
                    host = cfg.get_env_config(Constants.WEBSOCKET_HOST)
                    port = cfg.get_env_config(Constants.WEBSOCKET_PORT)
                    
                    WebSocketEventNotifier.hub_url = f"ws://{host}:{port}"
                    logger.info(f"[WebSocket] Attempting to connect to Hub at {WebSocketEventNotifier.hub_url}...")
                    
                    ws = websocket.WebSocketApp(
                        WebSocketEventNotifier.hub_url,
                        on_open=lambda ws: logger.info("[WebSocket] Successfully connected to Hub."),
                        on_message=lambda ws, msg: logger.debug(f"[WebSocket] Message received from Hub: {msg}"),
                        on_error=lambda ws, err: logger.error(f"[WebSocket] Hub Connection Error: {err}"),
                        on_close=lambda ws, status, msg: logger.info(f"[WebSocket] Hub Connection Closed: {status} - {msg}")
                    )
                    
                    WebSocketEventNotifier.ws = ws
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                    
                except Exception as e:
                    logger.error(f"[WebSocket] Client execution failure: {e}", exc_info=True)
                
                # Wait before attempting to reconnect
                logger.info("[WebSocket] Retrying connection in 5 seconds...")
                time.sleep(5)

        client_thread = threading.Thread(target=run_client, daemon=True)
        client_thread.start()
        return client_thread
