import json
import threading
from websocket_server import WebsocketServer
from src.utils.logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.exception.exception import WebSocketError

logger = LoggingConfig.setup_logging()

class WebSocketEventNotifier:
    """
    Class to handle real-time event notifications via WebSockets.
    """
    server = None
    clients = []

    @staticmethod
    def broadcast_event(payload):
        """
        Sends structured event data to all connected WebSocket clients.
        """
        if not WebSocketEventNotifier.server:
            logger.debug("[WebSocket] Server not initialized. Cannot broadcast.")
            raise WebSocketError("WebSocket Server not initialized. Cannot broadcast.")

        message = json.dumps(payload)
        logger.info(f"[WebSocket] Broadcasting event to {len(WebSocketEventNotifier.clients)} clients: {message[:100]}...")

        for client in WebSocketEventNotifier.clients:
            try:
                WebSocketEventNotifier.server.send_message(client, message)
            except Exception as e:
                logger.error(f"[WebSocket] Error sending message to client {client['id']}: {e}", exc_info=True)
                # Intentionally not raising here to allow broadcasting to remaining clients

        return True

    @staticmethod
    def start_websocket_server():
        """
        Starts the WebSocket server in a background thread.
        """
        def run_server():
            def new_client(client, server):
                logger.info(f"[WebSocket] New client connected: {client['id']}")
                WebSocketEventNotifier.clients.append(client)

            def client_left(client, server):
                logger.info(f"[WebSocket] Client disconnected: {client['id']}")
                if client in WebSocketEventNotifier.clients:
                    WebSocketEventNotifier.clients.remove(client)

            def message_received(client, server, message):
                logger.info(f"[WebSocket] Received message from client {client['id']}: {message}")

            try:
                host = cfg.get_env_config(Constants.WEBSOCKET_HOST)
                port = int(cfg.get_env_config(Constants.WEBSOCKET_PORT))
                
                server = WebsocketServer(port=port, host=host)
                server.set_fn_new_client(new_client)
                server.set_fn_client_left(client_left)
                server.set_fn_message_received(message_received)

                WebSocketEventNotifier.server = server
                logger.info(f"[WebSocket] Server successfully started on ws://{host}:{port}")
                server.run_forever()
            except Exception as e:
                logger.critical(f"[WebSocket] Failed to start WebSocket server: {e}", exc_info=True)
                raise WebSocketError(f"Failed to start WebSocket server: {e}")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
