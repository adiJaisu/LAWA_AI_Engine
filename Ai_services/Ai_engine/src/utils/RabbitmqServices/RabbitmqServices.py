import pika,socket,time
import json
import os
import sys
import threading
import queue
import msgpack
from typing import List, Dict, Any
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.Exception.Exception import InvalidInputException, RabbitMQConnectionError
from pika.exceptions import AMQPConnectionError, StreamLostError, ConnectionClosedByBroker
from src.constant.global_constant import VisionPipeline

logging_config = LoggingConfig()
logger = logging_config.setup_logging()


class RabbitMQService:
    """
    Thread-safe RabbitMQ Service using thread-local connections and channels.
    """

    def __init__(self, host: str, port: int, username: str, password: str, exchange_name: str, exchange_type: str):
        if not all(isinstance(arg, str) for arg in [host, username, password, exchange_name, exchange_type]):
            raise InvalidInputException("All parameters except port must be strings.")
        if not isinstance(port, int):
            raise InvalidInputException("Port must be an integer.")

        self.rabbitmq_host = host
        self.rabbitmq_port = port
        self.username = username
        self.password = password
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type

        self._thread_local = threading.local()
        self._lock = threading.Lock()

    def _ensure_connection(self):
        """
        Ensure RabbitMQ connection for current thread.
        Retries until MAX_CONNECTION_RETRY_DURATION is exceeded.
        """

        # Fast path
        if (
            hasattr(self._thread_local, "connection")
            and self._thread_local.connection
            and self._thread_local.connection.is_open
        ):
            return

        start_time = time.time()

        while (time.time() - start_time) < Constants.MAX_CONNECTION_RETRY_DURATION:

            if VisionPipeline.shutdown_event.is_set():
                logger.info(f"[{threading.current_thread().name}] Shutdown detected. Stopping RabbitMQ retries.")
                return

            try:
                credentials = pika.PlainCredentials(self.username, self.password)

                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=self.rabbitmq_host,
                        port=self.rabbitmq_port,
                        heartbeat=Constants.CONNECTION_HEARTBEAT,
                        blocked_connection_timeout=Constants.BLOCKED_CONNECTION_TIMEOUT,
                        credentials=credentials,
                    )
                )

                channel = connection.channel()
                channel.exchange_declare(
                    exchange=self.exchange_name,
                    exchange_type=self.exchange_type,
                    durable=True,
                )

                self._thread_local.connection = connection
                self._thread_local.channel = channel

                logger.info(f"[{threading.current_thread().name}] RabbitMQ connected successfully")
                return
                
            except socket.gaierror as e:
                logger.error(f"[{threading.current_thread().name}] DNS Resolution Failed for RabbitMQ host: {e}. Retrying in {Constants.CONNECTION_RETRY_DELAY} seconds...")
                elapsed = int(time.time() - start_time)
                VisionPipeline.shutdown_event.wait(Constants.CONNECTION_RETRY_DELAY)

            except AMQPConnectionError as e:
                logger.error(f"[{threading.current_thread().name}] RabbitMQ Connection Error: {e}. Retrying in {Constants.CONNECTION_RETRY_DELAY} seconds...")
                elapsed = int(time.time() - start_time)
                VisionPipeline.shutdown_event.wait(Constants.CONNECTION_RETRY_DELAY)

            except Exception as e:
                logger.error(f"[{threading.current_thread().name}] Something Unexpected happened: {e}. Retrying in {Constants.CONNECTION_RETRY_DELAY} seconds...")
                elapsed = int(time.time() - start_time)
                VisionPipeline.shutdown_event.wait(Constants.CONNECTION_RETRY_DELAY)

    def _get_channel(self):
        self._ensure_connection()
        channel = getattr(self._thread_local, "channel", None)
        if not channel or channel.is_closed:
            raise RabbitMQConnectionError("Channel not available")
        return channel


    def send_messages_batch(self, queue_name: str, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages to queue with declared exchange."""
        if not isinstance(queue_name, str):
            raise ValueError("queue_name must be a string.")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        try:
            channel = self._get_channel()
            payload = json.dumps(messages)
            channel.basic_publish(
                exchange=self.exchange_name,
                routing_key=queue_name,
                body=payload,
                properties=pika.BasicProperties(delivery_mode=1)
            )
            logger.debug(f"Size of one batch of messages: {sys.getsizeof(payload)} bytes")
            logger.debug(f"[{threading.current_thread().name}] Sent {len(messages)} messages in one batch to {queue_name}")

        except (StreamLostError, AMQPConnectionError) as e:
            logger.warning(f"[{threading.current_thread().name}] Failed to send messages batch (connection lost)")

        except Exception as e:
            logger.error(f"[{threading.current_thread().name}] Failed to send messages batch")

    def on_message(self, ch, method, properties, body):
        try:
            message = msgpack.unpackb(body, raw=False)
            VisionPipeline.global_rabbitmq_message_queue.put(message, timeout=2)

        except msgpack.FormatError:
            logger.warning("Invalid msgpack format in message, stopping consumer.")

        except queue.Full:
            logger.warning("Global queue is full..")

        except Exception as e:
            logger.error(f"Unexpected error processing message: {e}", exc_info=True)

    def receive_messages_batch(self, queue_name: str, prefetch_count: int = 50) -> None:
        """Consume messages with manual ack and prefetch limit."""
        if not isinstance(queue_name, str):
            raise InvalidInputException("queue_name must be a string.")

        try:
            channel = self._get_channel()

            channel.basic_qos(prefetch_count=prefetch_count)
            
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=self.on_message,
                auto_ack=True
            )

            logger.info(f"Consumer started on queue '{queue_name}' with prefetch={prefetch_count}")
            
            while not VisionPipeline.shutdown_event.is_set():
                self._thread_local.connection.process_data_events(time_limit=Constants.TEN_SEC)

        except (ConnectionClosedByBroker, StreamLostError, AMQPConnectionError, socket.gaierror) as e:
            logger.warning(f"[{threading.current_thread().name}] Connection lost/closed during consumption")

        except Exception as e:
            logger.error(f"[{threading.current_thread().name}] Failed to receive messages")

    def close_connection(self) -> None:
        """Close the RabbitMQ connection for current thread if open."""
        try:
            conn = getattr(self._thread_local, "connection", None)
            if not conn:
                return

            if conn.is_open:
                try:
                    conn.close()
                    logger.info(f"[{threading.current_thread().name}] RabbitMQ connection closed.")
                except (StreamLostError, ConnectionResetError):
                    logger.info(f"[{threading.current_thread().name}] RabbitMQ connection already closed by broker.")
            else:
                logger.debug(f"[{threading.current_thread().name}] RabbitMQ connection already closed.")

        except Exception as e:
            logger.warning(f"[{threading.current_thread().name}] Unexpected issue while closing RabbitMQ connection: {e}")

def initialize_rabbitmq_connection() -> RabbitMQService:
    """Factory to initialize and return RabbitMQService from env/config."""
    try:
        logger.info("Initializing RabbitMQ connection...")

        host = os.environ.get(Constants.RABBITMQ_HOST, cfg.get_env_config(Constants.RABBITMQ_HOST))
        port = int(os.environ.get(Constants.RABBITMQ_PORT, cfg.get_env_config(Constants.RABBITMQ_PORT)))
        username = os.environ.get(Constants.RABBITMQ_USERNAME, cfg.get_env_config(Constants.RABBITMQ_USERNAME))
        password = os.environ.get(Constants.RABBITMQ_PASSWORD, cfg.get_env_config(Constants.RABBITMQ_PASSWORD))
        exchange_name = os.environ.get(Constants.EXCHANGE_NAME, cfg.get_env_config(Constants.EXCHANGE_NAME))
        exchange_type = os.environ.get(Constants.EXCHANGE_TYPE, cfg.get_env_config(Constants.EXCHANGE_TYPE))

        service = RabbitMQService(
            host=host,
            port=port,
            username=username,
            password=password,
            exchange_name=exchange_name,
            exchange_type=exchange_type,
        )

        # Eager connection for main thread
        service._ensure_connection()
        logger.info(f"RabbitMQ connection ready on {host}:{port} using exchange '{exchange_name}'")

        return service
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQService: {e}", exc_info=True)
        raise RabbitMQConnectionError("Initialization failed")
