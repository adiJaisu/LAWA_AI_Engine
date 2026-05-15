import pika
import json
import threading
import datetime
import time
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.utils.video_processor import VideoEvidenceProcessor
from src.utils.logger import LoggingConfig
from src.utils.database import PostgresManager
from src.utils.websocket_notifier import WebSocketEventNotifier
from src.exception.exception import (
    EventManagerException,
    DatabaseConnectionError,
    VideoProcessingError,
    WebSocketError,
    MessageProcessingError
)

logger = LoggingConfig.setup_logging()

class EventManagerNode:
    def __init__(self):
        try:
            storage_path = cfg.get_env_config(Constants.EVIDENCE_STORAGE_PATH)
            self.processor = VideoEvidenceProcessor(base_storage_path=storage_path)
            self.db_manager = PostgresManager()

            self.rmq_host = cfg.get_env_config(Constants.RABBITMQ_HOST)
            self.rmq_user = cfg.get_env_config(Constants.RABBITMQ_USERNAME)
            self.rmq_pass = cfg.get_env_config(Constants.RABBITMQ_PASSWORD)
            self.exchange_name = cfg.get_env_config(Constants.EXCHANGE_NAME)
            self.exchange_type = cfg.get_env_config(Constants.EXCHANGE_TYPE)

            # List of queues to monitor
            self.event_queues = [
                Constants.LOITERING_DETECTION_EVENT_QUEUE,
                Constants.TAILGATE_DETECTION_EVENT_QUEUE,
                Constants.IN_OUT_PERSON_COUNT_EVENT_QUEUE,
                Constants.TRAIN_ARRIVAL_DEPART_MONITOR_EVENT_QUEUE,
                Constants.CROWD_DENSITY_EVENT_QUEUE,
                Constants.PERSON_ENTERED_INSIDE_TRAIN_EVENT_QUEUE,
                Constants.RESTROOM_PERSON_TRACKING_EVENT_QUEUE,
                Constants.PERSON_COUNT_INSIDE_COMPARTMENT_EVENT_QUEUE,
                Constants.QUEUE_MANAGEMENT_EVENT_QUEUE,
                Constants.BIRD_EYE_VIEW_EVENT_QUEUE
            ]
            
            # Initialize WebSocket Client in the background
            WebSocketEventNotifier.initialize_websocket_client()
            logger.info("EventManagerNode initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize EventManagerNode: {e}", exc_info=True)
            raise

    def start_consuming(self, queue_name):
        """Dedicated consumer thread for a specific queue."""
        logger.info(f"[*] Starting listener for {queue_name}...")
        
        try:
            credentials = pika.PlainCredentials(self.rmq_user, self.rmq_pass)
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=self.rmq_host, 
                credentials=credentials,
                heartbeat=600
            ))
            
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.exchange_declare(exchange=self.exchange_name, exchange_type=self.exchange_type, durable=True)
            channel.queue_bind(exchange=self.exchange_name, queue=queue_name, routing_key=queue_name)

            def callback(ch, method, properties, body):
                try:
                    try:
                        event_batch = json.loads(body)
                    except json.JSONDecodeError as e:
                        raise MessageProcessingError(f"Failed to decode JSON from RabbitMQ message: {e}")

                    if isinstance(event_batch, dict):
                        event_batch = [event_batch]

                    if not event_batch:
                        logger.warning(f"[{queue_name}] Received empty event batch. Ignoring.")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    meta = event_batch[0].get("camera_metadata", {})
                    frame_meta = event_batch[0].get("frame_metadata", {})
                    
                    camera_id = meta.get("camera_id", "Unknown_Camera")
                    usecase = frame_meta.get("usecase_name", "Unknown_Usecase")
                    
                    frames = []
                    for msg in event_batch:
                        f = msg.get("frame_metadata", {}).get("frame") or msg.get("frame_metadata", {}).get("output_image_data")
                        if f:
                            if isinstance(f, list): 
                                frames.extend(f)
                            else: 
                                frames.append(f)

                    logger.debug(f"[EVENT] Received {len(frames)} frames for {usecase} from {camera_id}.")

                    # Generate video evidence locally
                    video_path = self.processor.process_frames(
                        camera_id=camera_id,
                        usecase_name=usecase,
                        frames_base64=frames
                    )

                    if video_path:
                        logger.info(f"[DONE] Evidence video saved locally: {video_path}")
                        
                        camera_asset_id = meta.get("camera_asset_id", camera_id)
                        location_id = meta.get("location_id", "Unknown")
                        usecase_id = frame_meta.get("usecase_id", 0)
                        object_name = frame_meta.get("object_name", "Unknown")
                        cluster = frame_meta.get("cluster", usecase)
                        request_status = 1
                        event_time = datetime.datetime.now().isoformat()
                        
                        # Build standard event payload using Constants
                        event_payload = {
                            Constants.CAMERA_ID: camera_id,
                            Constants.CAMERA_ASSET_ID: camera_asset_id,
                            Constants.SENSOR_EVENT_TIME: event_time,
                            Constants.USECASE_ID: usecase_id,
                            Constants.SENSOR_EVENT_TYPE: object_name,
                            Constants.EVIDENCE_STORAGE_PATH: video_path,
                            Constants.REQUEST_STATUS: request_status,
                            Constants.META_LOCATION_ID: location_id,
                            Constants.SENSOR_EVENT_DATA: Constants.DETECTION_DICT.get(cluster, "Event Detected")
                        }
                        
                        # Save to PostgreSQL
                        event_id = self.db_manager.insert_event(
                            camera_id=camera_id, 
                            usecase_name=usecase, 
                            evidence_path=video_path, 
                            event_data=event_payload
                        )
                        
                        if event_id:
                            event_payload["event_id"] = event_id

                        # Broadcast via WebSocket
                        WebSocketEventNotifier.broadcast_event(event_payload)
                    else:
                        logger.error(f"[ERROR] Failed to process video evidence for {usecase} on {camera_id}.")

                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except MessageProcessingError as e:
                    logger.error(f"[{queue_name}] Message processing failed: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                except VideoProcessingError as e:
                    logger.error(f"[{queue_name}] Video generation failed: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                except DatabaseConnectionError as e:
                    logger.error(f"[{queue_name}] Database connection/insertion failed: {e}")
                    # Requeue so we don't lose events when DB is temporarily down
                    time.sleep(5)
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                except WebSocketError as e:
                    logger.error(f"[{queue_name}] WebSocket broadcast failed: {e}")
                    # Don't nack here; DB insertion was successful, so event shouldn't be lost completely
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logger.error(f"[{queue_name}] Unexpected callback failure: {e}", exc_info=True)
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            logger.critical(f"[{queue_name}] RabbitMQ Connection error: {e}")
        except Exception as e:
            logger.critical(f"[{queue_name}] Critical failure in consumer thread: {e}", exc_info=True)

    def run(self):
        """Starts all consumer threads."""
        logger.info(f"[SYSTEM] Event Manager starting with {len(self.event_queues)} consumer threads...")
        threads = []
        for q in self.event_queues:
            t = threading.Thread(target=self.start_consuming, args=(q,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("[SYSTEM] Shutting down Event Manager gracefully due to KeyboardInterrupt.")

if __name__ == "__main__":
    manager = EventManagerNode()
    manager.run()
