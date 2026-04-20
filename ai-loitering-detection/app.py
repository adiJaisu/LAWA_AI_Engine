import asyncio
import os
import time
import threading
from src.constant.global_constant import VisionPipeline
from src.utils.RabbitmqServices.RabbitmqConsumer import RabbitMQConsumerManager
from src.utils.RabbitmqServices.RabbitmqServices import initialize_rabbitmq_connection
from src.utils.Logger import LoggingConfig
from src.utils.LogCleaner import start_log_cleanup_thread
from src.utils.ConfigReader import cfg
from src.utils.GPUdevicemanager import GPUManager
from src.constant.constants import Constants
from src.Exception.Exception import ModelLoadingError, RabbitMQConnectionError
from src.main import process_vision_stream

logging_config = LoggingConfig()
logger = logging_config.setup_logging()
VisionPipeline.queue_name = os.environ.get(Constants.QUEUE_NAME, Constants.LOITERING_DETECTION_QUEUE)
num_consumer = int(os.environ.get(Constants.NUM_CONSUMER_THREADS, Constants.FOUR))
rabbitmq_consumer = RabbitMQConsumerManager(VisionPipeline.queue_name, num_consumer)

def initialize_detectors() -> None:
    """
    Initializes the loitering detection model and GPU manager.

    Configuration priority:
    1. Environment variables
    2. Config file values as fallback

    Required configurations:
    - LOITERING_DETECTION_MODEL_PATH: Path to the loitering detection model
    - NUMBERS_OF_WORKER: Number of worker processes per GPU

    Raises:
        ModelLoadingError: If model initialization fails
        ValueError: If configuration values are invalid or missing
    """
    try:
        logger.info("Initializing loitering detection model and GPU manager...")
        model_path: str = os.environ.get(
            Constants.LOITERING_DETECTION_MODEL_PATH,
            cfg.get_value_config(Constants.LOITERING_DETECTION, Constants.LOITERING_DETECTION_MODEL_PATH)
        )
        number_of_worker: int = int(os.environ.get(
            Constants.NUMBERS_OF_WORKER,
            cfg.get_env_config(Constants.NUMBERS_OF_WORKER)
        ))

        if not model_path or not number_of_worker:
            raise ValueError("Missing required model path or worker configuration.")

        VisionPipeline.gpu_manager = GPUManager(
            num_workers_per_gpu=number_of_worker,
            model_path=model_path,
            secondary_model_path=None
        )
        time.sleep(Constants.FIVE_SEC)
        logger.info("Loitering detection model and GPU manager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize loitering detection model or GPU manager: {str(e)}", exc_info=True)
        raise ModelLoadingError(f"Failed to initialize detectors: {str(e)}")
    
async def execute_VisionAiPipeline() -> None:
    """
    Executes the Vision AI loitering detection pipeline.

    Steps:
    1. Initializes loitering detection model and GPU manager.
    2. Establishes RabbitMQ connection.
    3. Starts vision stream processing.

    Handles errors and logs all major events.

    Raises:
        ModelLoadingError: If model or GPU manager initialization fails.
        RabbitMQConnectionError: If RabbitMQ connection fails.
        Exception: For any other pipeline errors.
    """
    try:
        logger.info("Launching Vision AI loitering detection pipeline...")
        initialize_detectors()
        VisionPipeline.rabbitmq_service = initialize_rabbitmq_connection()
        await process_vision_stream()
        logger.info("Vision AI pipeline execution completed successfully.")
    except (ModelLoadingError, RabbitMQConnectionError) as e:
        logger.critical(f"Critical pipeline initialization error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"Unexpected pipeline execution failure: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Start RabbitMQ consumer threads
        threads = rabbitmq_consumer.start_multiple_consumer_threads()

        # logs cleaner thread 
        start_log_cleanup_thread()
        # Start async processing (which uses global queue)
        asyncio.run(execute_VisionAiPipeline())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cleaning up resources and exiting.")
    except Exception as e:
        logger.critical(f"Fatal error during pipeline execution: {str(e)}", exc_info=True)
        exit(1)
