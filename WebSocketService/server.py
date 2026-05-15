import asyncio
import websockets
import json
import logging
import os
from src.utils.logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants

# Initialize Logger
logger = LoggingConfig.setup_logging()

CLIENTS = set()

async def handler(websocket):
    """
    Handles incoming WebSocket connections and broadcasts messages.
    """
    CLIENTS.add(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected. Total clients: {len(CLIENTS)}")
    
    try:
        async for message in websocket:
            logger.info(f"[HUB] Received message from client {client_id}: {message[:200]}...")
            
            # Broadcast to all other connected clients
            if len(CLIENTS) > 1:
                logger.info(f"[HUB] Broadcasting to {len(CLIENTS) - 1} other clients.")
                broadcast_tasks = [
                    asyncio.create_task(client.send(message))
                    for client in CLIENTS
                    if client != websocket
                ]
                if broadcast_tasks:
                    await asyncio.wait(broadcast_tasks)
            else:
                logger.debug(f"[HUB] No other clients connected to receive broadcast.")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} connection closed.")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    finally:
        if websocket in CLIENTS:
            CLIENTS.remove(websocket)
        logger.info(f"Client {client_id} disconnected. Total clients: {len(CLIENTS)}")

async def main():
    host = cfg.get_env_config(Constants.WEBSOCKET_HOST)
    port = int(cfg.get_env_config(Constants.WEBSOCKET_PORT))
    
    async with websockets.serve(handler, host, port):
        logger.info(f"WebSocket Hub successfully started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("WebSocket Hub shutting down.")
