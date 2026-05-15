import os
import time
import json
import pika
import numpy as np
from src.constant.global_constant import VisionPipeline
from src.utils.Logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.utils.GPUdevicemanager import GPUManager
from src.constant.constants import Constants
from src.Exception.Exception import ModelLoadingError
import threading
from src.utils.Logger import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.setup_logging()

VisionPipeline.shutdown_event = threading.Event()
VisionPipeline.shutdown_lock = threading.Lock()

def initialize_classification_models() -> None:
    """
    Initializes the centralized AI inference models and GPU manager.
    """
    try:
        logger.info("Initializing centralized model server and GPU manager...")
        model_path: str = os.environ.get(
            Constants.AI_INFERENCE_CLASSIFICATION_MODEL_PATH,
            cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.AI_INFERENCE_CLASSIFICATION_MODEL_PATH)
        )
        number_of_worker: int = int(os.environ.get(
            Constants.NUMBERS_OF_WORKER,
            cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.NUMBERS_OF_WORKER)
        ))

        if not model_path:
            logger.warning("No MODEL_PATH provided. Falling back to default YOLO model.")
            model_path = "yolov8n-cls.pt"

        VisionPipeline.gpu_manager = GPUManager(
            num_workers_per_gpu=number_of_worker,
            model_path=model_path,
            secondary_model_path=None
        )
        time.sleep(5)
        logger.info("AI Inference model and GPU manager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize centralized AI inference service: {str(e)}", exc_info=True)
        raise ModelLoadingError(f"Failed to initialize classification models: {str(e)}")

def run_inference_task(model, secondary_model, task_data):
    """
    Executed by a GPUWorker inside GPUManager.
    """
    ch, method, props, payload, frame = task_data
    method_name = payload.get("method")
    kwargs = payload.get("kwargs", {})
    
    response = {}
    try:
        if method_name == "make_prediction_with_classification":
            results = model.make_prediction_with_classification(frame, **kwargs)
        else:
            raise ValueError(f"Unknown method {method_name}")

        if results and hasattr(results[0], "probs"):
            # Extract classification probabilities
            top1_idx = int(results[0].probs.top1)
            top1_conf = float(results[0].probs.top1conf)
            # You can also get names mapping if needed, but for now top class is enough
            response = {"top1_idx": top1_idx, "top1_conf": top1_conf}
        else:
            response = {"top1_idx": None, "top1_conf": 0.0}
    except Exception as e:
        logger.error(f"Inference error: {e}")
        response = {"error": str(e)}
        
    # Delete shared memory file
    try:
        shm_path = payload.get("shm_path")
        if shm_path and os.path.exists(shm_path):
            os.remove(shm_path)
    except Exception as e:
        logger.error(f"Failed to delete shm file: {e}")

    # Publish response back to RabbitMQ using threadsafe callback
    try:
        cb = lambda: ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=json.dumps(response)
        )
        ch.connection.add_callback_threadsafe(cb)
        
        # Acknowledge the original request
        ch.connection.add_callback_threadsafe(lambda: ch.basic_ack(delivery_tag=method.delivery_tag))
    except Exception as e:
        logger.error(f"Failed to send RPC response: {e}")

def on_request(ch, method, props, body):
    """
    RabbitMQ consumer callback for incoming RPC requests.
    """
    try:
        payload = json.loads(body)
        shm_path = payload["shm_path"]
        shape = payload["shape"]
        dtype = payload["dtype"]
        
        # Read frame from shared memory
        with open(shm_path, "rb") as f:
            raw_bytes = f.read()
            
        frame = np.frombuffer(raw_bytes, dtype=np.dtype(dtype)).reshape(shape)
        
        task_data = (ch, method, props, payload, frame)
        
        # Assign to GPU worker
        # GPUManager.assign_task expects (func, args...)
        # But we only need to pass func and task_data. Looking at GPUManager code, it passes
        # func(model, secondary_model, validated_msg_with_frames_and_metadatas)
        # So task_data will map to validated_msg_with_frames_and_metadatas
        assigned = VisionPipeline.gpu_manager.assign_task(run_inference_task, task_data)
        if not assigned:
            logger.debug("Workers busy, retrying assignment...")
            VisionPipeline.gpu_manager.retry_assign_task(run_inference_task, task_data)
            
    except Exception as e:
        logger.error(f"Error processing RPC request: {e}")
        # Send error response
        if props.reply_to:
            err_response = {"error": f"Server parse error: {str(e)}"}
            cb = lambda: ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                properties=pika.BasicProperties(correlation_id=props.correlation_id),
                body=json.dumps(err_response)
            )
            ch.connection.add_callback_threadsafe(cb)
        ch.basic_ack(delivery_tag=method.delivery_tag) # Ack so it's not requeued infinitely

def execute_AiInferenceClassification() -> None:
    """
    Executes the Ai Inference Classification pipeline.
    """
    try:
        logger.info("Launching AI Inference Classification...")
        initialize_classification_models()
        
        host = os.environ.get(Constants.RABBITMQ_HOST, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_HOST))
        port = int(os.environ.get(Constants.RABBITMQ_PORT, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PORT)))
        username = os.environ.get(Constants.RABBITMQ_USERNAME, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_USERNAME))
        password = os.environ.get(Constants.RABBITMQ_PASSWORD, cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.RABBITMQ_PASSWORD))
        ai_inference_classification_queue = str(os.environ.get(
            Constants.AI_INFERENCE_CLASSIFICATION_QUEUE,
            cfg.get_value_config(Constants.DEFAULT_ENVIRONMENT, Constants.AI_INFERENCE_CLASSIFICATION_QUEUE)))
        credentials = pika.PlainCredentials(username, password)
        parameters = pika.ConnectionParameters(host=host, port=port, credentials=credentials, heartbeat=600)
        
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        channel.queue_declare(queue=ai_inference_classification_queue)
        # Fair dispatch: give one message to a worker at a time
        channel.basic_qos(prefetch_count=1)
        
        # Note: auto_ack is False because we ack AFTER the GPU finishes
        channel.basic_consume(queue=ai_inference_classification_queue, on_message_callback=on_request)
        
        logger.info("Awaiting RPC requests...")
        channel.start_consuming()

    except Exception as e:
        logger.critical(f"Unexpected pipeline execution failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        execute_AiInferenceClassification()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cleaning up resources and exiting.")
    except Exception as e:
        logger.critical(f"Fatal error during pipeline execution: {str(e)}", exc_info=True)
        exit(1)
