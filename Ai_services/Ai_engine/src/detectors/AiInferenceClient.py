import json
import uuid
import os
import time
import threading
import pika
import numpy as np
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

logger = LoggingConfig().setup_logging()

class RemoteTensor:
    def __init__(self, data):
        self.data = data
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self.data)

class RemoteBoxes:
    def __init__(self, xyxy, ids=None):
        self.xyxy = RemoteTensor(xyxy)
        self.id = RemoteTensor(ids) if ids is not None else None

class RemoteYoloResult:
    def __init__(self, xyxy, ids):
        self.boxes = RemoteBoxes(xyxy, ids)


class AiInferenceClient:
    """
    Client to send frames to the centralized AI Inference Service via RabbitMQ RPC and Shared Memory.
    Designed to mimic the ObjectDetector interface so business logic remains unchanged.
    Uses thread-local connections to safely support ThreadPoolExecutor.
    """

    def __init__(self):
        self.local = threading.local()
        self.host = os.environ.get(Constants.RABBITMQ_HOST, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_HOST))
        self.port = int(os.environ.get(Constants.RABBITMQ_PORT, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PORT)))
        self.username = os.environ.get(Constants.RABBITMQ_USERNAME, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_USERNAME))
        self.password = os.environ.get(Constants.RABBITMQ_PASSWORD, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PASSWORD))
        self.ai_inference_queue = str(os.environ.get(Constants.AI_INFERENCE_QUEUE, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.AI_INFERENCE_QUEUE)))
        self.shm_dir = "/dev/shm"

    def _get_connection(self):
        """Get or create a thread-local RabbitMQ connection and callback queue."""
        if not hasattr(self.local, 'connection') or self.local.connection.is_closed:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials, heartbeat=600)
            self.local.connection = pika.BlockingConnection(parameters)
            self.local.channel = self.local.connection.channel()
            
            # Declare a unique callback queue for this thread
            result = self.local.channel.queue_declare(queue='', exclusive=True)
            self.local.callback_queue = result.method.queue
            
            # Setup consuming on the callback queue
            self.local.channel.basic_consume(
                queue=self.local.callback_queue,
                on_message_callback=self._on_response,
                auto_ack=True
            )
            self.local.response = None
            self.local.corr_id = None
            
        return self.local.channel

    def _on_response(self, ch, method, props, body):
        if self.local.corr_id == props.correlation_id:
            self.local.response = body

    def _call_rpc(self, payload: dict, timeout: int = Constants.TEN_SEC) -> dict:
        channel = self._get_connection()
        self.local.response = None
        self.local.corr_id = str(uuid.uuid4())
        
        channel.basic_publish(
            exchange='',
            routing_key=self.ai_inference_queue,
            properties=pika.BasicProperties(
                reply_to=self.local.callback_queue,
                correlation_id=self.local.corr_id,
            ),
            body=json.dumps(payload)
        )
        
        # Wait for the response
        start_time = time.time()
        while self.local.response is None:
            self.local.connection.process_data_events(time_limit=0.01)
            # Add a small sleep to prevent 100% CPU usage while waiting
            time.sleep(0.001)
            if time.time() - start_time > timeout:
                logger.error(f"RPC call timed out after {timeout} seconds for queue {self.ai_inference_queue}")
                return {"error": "Timeout", "xyxy": [], "ids": None}
            
        response_data = json.loads(self.local.response)
        return response_data

    def _write_shm(self, frame: np.ndarray) -> str:
        file_id = str(uuid.uuid4())
        shm_path = os.path.join(self.shm_dir, f"frame_{file_id}.raw")
        with open(shm_path, "wb") as f:
            f.write(frame.tobytes())
        return shm_path

    def make_prediction(self, frame: np.ndarray, classes_id: list[int], confidence: float, img_size: int = 640) -> list | None:
        """Mimics ObjectDetector.make_prediction"""
        try:
            shm_path = self._write_shm(frame)
            
            payload = {
                "method": "make_prediction",
                "shm_path": shm_path,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "kwargs": {
                    "classes_id": classes_id,
                    "confidence": confidence,
                    "img_size": img_size
                }
            }
            
            response = self._call_rpc(payload)
            
            if response.get("error"):
                logger.error(f"Inference Server Error: {response['error']}")
                return None
                
            xyxy = response.get("xyxy", [])
            ids = response.get("ids", None)
            
            if not xyxy:
                return []
                
            return [RemoteYoloResult(xyxy, ids)]
            
        except Exception as e:
            logger.error(f"RPC Prediction Error: {e}")
            return None

    def make_prediction_with_tracking(self, frame: np.ndarray, classes_id: list[int], confidence: float, inference_image_size: int = 640, tracker: str = "bytetrack.yaml", iou: float = Constants.ZERO_POINT_FIVE) -> list:
        """Mimics ObjectDetector.make_prediction_with_tracking"""
        try:
            shm_path = self._write_shm(frame)
            
            payload = {
                "method": "make_prediction_with_tracking",
                "shm_path": shm_path,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "kwargs": {
                    "classes_id": classes_id,
                    "confidence": confidence,
                    "inference_image_size": inference_image_size,
                    "tracker": tracker,
                    "iou": iou
                }
            }
            
            response = self._call_rpc(payload)
            
            if response.get("error"):
                logger.error(f"Inference Server Error: {response['error']}")
                return []
                
            xyxy = response.get("xyxy", [])
            ids = response.get("ids", None)
            
            if not xyxy:
                return []
                
            return [RemoteYoloResult(xyxy, ids)]
            
        except Exception as e:
            logger.error(f"RPC Prediction Tracking Error: {e}")
            return []



class AiInferenceClassificationClient:
    """
    Client to send frames to the centralized AI Inference Classification Service via RabbitMQ RPC and Shared Memory.
    Designed to mimic the classification model interface so business logic remains unchanged.
    Uses thread-local connections to safely support ThreadPoolExecutor.
    """

    def __init__(self):
        self.local = threading.local()
        self.host = os.environ.get(Constants.RABBITMQ_HOST, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_HOST))
        self.port = int(os.environ.get(Constants.RABBITMQ_PORT, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PORT)))
        self.username = os.environ.get(Constants.RABBITMQ_USERNAME, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_USERNAME))
        self.password = os.environ.get(Constants.RABBITMQ_PASSWORD, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PASSWORD))
        self.ai_inference_classification_queue = str(os.environ.get(Constants.AI_INFERENCE_CLASSIFICATION_QUEUE, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.AI_INFERENCE_CLASSIFICATION_QUEUE)))
        self.shm_dir = "/dev/shm"

    def _get_connection(self):
        """Get or create a thread-local RabbitMQ connection and callback queue."""
        if not hasattr(self.local, 'connection') or self.local.connection.is_closed:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials, heartbeat=600)
            self.local.connection = pika.BlockingConnection(parameters)
            self.local.channel = self.local.connection.channel()
            
            # Declare a unique callback queue for this thread
            result = self.local.channel.queue_declare(queue='', exclusive=True)
            self.local.callback_queue = result.method.queue
            
            # Setup consuming on the callback queue
            self.local.channel.basic_consume(
                queue=self.local.callback_queue,
                on_message_callback=self._on_response,
                auto_ack=True
            )
            self.local.response = None
            self.local.corr_id = None
            
        return self.local.channel

    def _on_response(self, ch, method, props, body):
        if self.local.corr_id == props.correlation_id:
            self.local.response = body

    def _call_rpc(self, payload: dict, timeout: int = Constants.TEN_SEC) -> dict:
        channel = self._get_connection()
        self.local.response = None
        self.local.corr_id = str(uuid.uuid4())
        
        channel.basic_publish(
            exchange='',
            routing_key=self.ai_inference_classification_queue,
            properties=pika.BasicProperties(
                reply_to=self.local.callback_queue,
                correlation_id=self.local.corr_id,
            ),
            body=json.dumps(payload)
        )
        
        # Wait for the response
        start_time = time.time()
        while self.local.response is None:
            self.local.connection.process_data_events(time_limit=0.01)
            # Add a small sleep to prevent 100% CPU usage while waiting
            time.sleep(0.001)
            if time.time() - start_time > timeout:
                logger.error(f"RPC call timed out after {timeout} seconds for queue {self.ai_inference_classification_queue}")
                return {"error": "Timeout", "predictions": []}
            
        response_data = json.loads(self.local.response)
        return response_data

    def _write_shm(self, frame: np.ndarray) -> str:
        file_id = str(uuid.uuid4())
        shm_path = os.path.join(self.shm_dir, f"frame_{file_id}.raw")
        with open(shm_path, "wb") as f:
            f.write(frame.tobytes())
        return shm_path

    def make_prediction_with_classification(self, frame: np.ndarray, confidence: float, img_size: int = 640) -> list:
        """Mimics ObjectDetector.make_prediction_with_classification"""
        try:
            shm_path = self._write_shm(frame)
            
            payload = {
                "method": "make_prediction_with_classification",
                "shm_path": shm_path,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "kwargs": {
                    "confidence": confidence,
                    "img_size": img_size
                }
            }
            
            response = self._call_rpc(payload)
            
            if response.get("error"):
                logger.error(f"Inference Server Error: {response['error']}")
                return []
                
            predictions = response.get("predictions", [])
            
            if not predictions:
                return []
            
            self.logger.info(f"Received classification predictions: {predictions}")    
            return [{"predictions": predictions}]
        
            
        except Exception as e:
            logger.error(f"RPC Classification Prediction Error: {e}")
            return []
