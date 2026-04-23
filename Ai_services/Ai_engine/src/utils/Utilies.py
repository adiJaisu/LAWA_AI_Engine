from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.utils.Streamhandler_validation.Streamhandler_validation import Validator
from src.utils.Logger import LoggingConfig,log_time
from src.utils.ConfigReader import cfg
from typing import List,Optional,Tuple,Dict,Any
from concurrent.futures import ThreadPoolExecutor
from src.detectors.ObjectDetector import ObjectDetector
import numpy as np
import cv2
import os
import copy
import turbojpeg
logger = LoggingConfig().setup_logging()

# import turbojpeg
try:
    jpeg = turbojpeg.TurboJPEG()
except RuntimeError:
    logger.warning("TurboJPEG C-library not found. Falling back to OpenCV natively for JPEG decoding.")
    jpeg = None

jpeg_decoder_executer = ThreadPoolExecutor(os.cpu_count())

class Utility:
    @staticmethod
    def check_alert_for_batch(batch):
        """ Check alert for Given batch """
        for item in batch:
            alert = item[Constants.FRAME_METADATA][Constants.ALERT]
            if alert:
                return True
        return False
    
    @staticmethod
    def check_vest_color(img_crop):
        img_crop_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        total_pixels = img_crop_hsv.shape[0] * img_crop_hsv.shape[1]
        if total_pixels == Constants.ZERO:
            return None, "none", 0.0

        color_match_info = {}
        for color_name, (lower, upper) in Constants.color_ranges.items():
            mask = cv2.inRange(img_crop_hsv, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            percentage = (count / total_pixels) * Constants.ONE_HUNDRED
            color_match_info[color_name] = (percentage, mask)

        best_color = max(color_match_info, key=lambda c: color_match_info[c][0])
        best_percent = color_match_info[best_color]
        return best_color, best_percent
    
    @staticmethod
    def split_into_sub_batch(validated_msg_with_frames_and_metadatas:List[Dict]):
        """Split one batch of frames and metadata into three sub-batches."""
        batch_split_ratio = int(cfg.get_value_config(Constants.INTRUSION_DETECTION, Constants.SUB_BATCH_COUNT))
        ten_minus = Constants.TEN - batch_split_ratio

        return (
            (validated_msg_with_frames_and_metadatas[:batch_split_ratio]),
            (validated_msg_with_frames_and_metadatas[batch_split_ratio:ten_minus]),
            (validated_msg_with_frames_and_metadatas[ten_minus:])
        )
    
    @staticmethod
    @log_time("Time Taken to complete parallel decoding of batch's frames", True)
    def parallel_decode_jpegs(validated_metadatas: List[dict], max_workers: int = 5) -> List[Optional[dict]]:
        """Return list of messages with decoded frames (np.ndarray in BGR), preserving input order."""
        if not validated_metadatas:
            return []

        results_iter = jpeg_decoder_executer.map(Utility._decode_one_jpeg, validated_metadatas)
        return list(results_iter)

    @staticmethod
    def _decode_one_jpeg(msg: dict) -> Optional[dict]:
        """Decode a single base64-encoded JPEG frame inside the message."""
        try:
            msg_copy = copy.deepcopy(msg)

            frame_buffer = msg_copy[Constants.FRAME_METADATA][Constants.FRAME]
            
            if jpeg is not None:
                decoded_frame_buffer = jpeg.decode(frame_buffer)
            else:
                nparr = np.frombuffer(frame_buffer, np.uint8)
                decoded_frame_buffer = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            msg_copy[Constants.FRAME_METADATA][Constants.FRAME] = decoded_frame_buffer
            return msg_copy

        except Exception as e:
            logger.error(f"[VISION STREAM] JPEG decode error: {e}")
            return None

    @staticmethod
    def decode_frame(validated_metadatas: List[dict]) -> List[Optional[dict]]:
        """Decode one batch of frames in parallel using multiple threads."""
        messages_with_decoded_frame = Utility.parallel_decode_jpegs(
            validated_metadatas,
            max_workers=int(cfg.get_env_config(Constants.DECODE_WORKERS))
        )
        return messages_with_decoded_frame

    @staticmethod
    @log_time("Time Taken to preprocess one batch of message", True)
    def preprocess_messages(messages: List[dict], validator: Validator) -> List[Dict]:
        """Preprocess one batch of messages for detection."""
        validated_metadata = [validator.validate_message(msg) for msg in messages]
        if any(valMsg is None for valMsg in validated_metadata):
            return None
        validated_msg_with_frames_and_metadatas = Utility.decode_frame(validated_metadata)
        return validated_msg_with_frames_and_metadatas

    
    @staticmethod
    def handle_shutdown_signal(signum,frame):
        """Signal handler to gracefully exit the process."""
        logger.info(f"Received shutdown signal: {signum}. Cleaning up...")
        with VisionPipeline.shutdown_lock:
            VisionPipeline.shutdown_event.set()  


class GpuManagerUtils:

    @staticmethod
    def get_primary_and_secondary_model(primary_model_path:str = None,secondary_model_path:str = None,device = None):
        """ helps to identify given path and load apropriate models """

        primary_model,secondary_model = None,None
        if primary_model_path is not None:
            primary_model = ObjectDetector(model_path=primary_model_path, device=device)
                
        if secondary_model_path is not None:
            secondary_model = ObjectDetector(model_path=secondary_model_path, device=device)

        return primary_model , secondary_model



class ResourcesCleaner:

    def shutdown_gpu_workers(self):
        """ Shutdown gpu worker gracefully"""

        try:
            if hasattr(VisionPipeline, Constants.GPU_MANAGER) and VisionPipeline.gpu_manager:
                logger.info("[INFO] Shutting down GPU/CPU workers...")
                
                # Use the shutdown method from the fixed GPU manager
                if hasattr(VisionPipeline.gpu_manager, Constants.SHUTDOWN):
                    VisionPipeline.gpu_manager.shutdown()
                else:
                    # Fallback for existing implementation
                    for i, worker_dict in enumerate(VisionPipeline.gpu_manager.workers):
                        try:
                            worker = worker_dict[Constants.WORKER]
                            if hasattr(worker, Constants.THREAD) and worker.thread and worker.thread.is_alive():
                                logger.info(f"[INFO] Waiting for GPU/CPU worker {i+1} to finish...")
                                worker.thread.join(timeout=Constants.TEN_SEC)
                                
                                if worker.thread.is_alive():
                                    logger.info(f"[WARNING] GPU/CPU worker {i+1} did not shut down gracefully within timeout")
                                else:
                                    logger.info(f"[INFO] GPU/CPU worker {i+1} shut down successfully")
                        except Exception as e:
                            logger.error(f"Error waiting for GPU/CPU worker {i+1}: {e}")
                
                logger.info("[INFO] All GPU workers shutdown complete")
                
        except Exception as e:
            logger.error(f"Error during GPU workers shutdown: {e}")

    def shutdown_rabbitMQ_consumers(self):
        """ Shut down RabbitMQ consumers gracefully"""

        try:
            if hasattr(VisionPipeline, 'rabbitmq_threads'):
                logger.info("[INFO] Waiting for RabbitMQ consumer threads to exit...")
                for i, t in enumerate(VisionPipeline.rabbitmq_threads):
                    if t.is_alive():
                        t.join(timeout=2)
                        if t.is_alive():
                            logger.info(f"[WARNING] RabbitMQ thread {t.name} did not shut down cleanly.")
                        else:
                            logger.info(f"[INFO] RabbitMQ thread {t.name} shut down successfully.")
        except Exception as e:
            logger.error(f"Error during RabbitMQ thread shutdown: {e}")

        logger.info("[INFO] Graceful shutdown sequence completed")
        
    def close_rabbitMQ_connection(self):
        """ Close rabbitMQ connection gracefully"""

        try:
            if hasattr(VisionPipeline, Constants.RABBITMQ_SERVICE) and VisionPipeline.rabbitmq_service:
                logger.info("[INFO] Closing RabbitMQ connection...")
                VisionPipeline.rabbitmq_service.close_connection()
                logger.info("[INFO] RabbitMQ connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")


    # def close_database_connection(self):
    #     """ Close open DataBase connection"""

    #     try:
    #         if (hasattr(VisionPipeline, Constants.DB_CONNECTION) and 
    #             VisionPipeline.db_connection is not None):
                
    #             logger.info("[INFO] Closing database connection...")
                
    #             # Close cursor first
    #             if (hasattr(VisionPipeline, Constants.CURSOR) and 
    #                 VisionPipeline.cursor is not None):
    #                 try:
    #                     VisionPipeline.cursor.close()
    #                     logger.info("[INFO] Database cursor closed")
    #                 except Exception as e:
    #                     logger.error(f"Error closing database cursor: {e}")
                
    #             # Then close connection
    #             if hasattr(VisionPipeline.db_connection, Constants.IS_CONNECTED) and VisionPipeline.db_connection.is_connected():
    #                 VisionPipeline.db_connection.close()
    #                 logger.info("[INFO] Database connection closed successfully")
    #             else:
    #                 logger.info("[INFO] Database connection was already closed")
                    
    #     except Exception as e:
    #         logger.error(f"Error closing database connection: {e}")
        

    async def cleanup_resources(self):
        """Centralized cleanup function with proper error handling and ordering"""

        # Step 1: Signal shutdown to all componentslog_worker_status
        try:
            with VisionPipeline.shutdown_lock:
                VisionPipeline.shutdown_event.set()
            logger.info("[INFO] Shutdown signal set for all components")
        except Exception as e:
            logger.error(f"Error setting shutdown signal: {e}")

        # Step 2: Gracefully shutdown GPU workers
        self.shutdown_gpu_workers()
        # Step 3: Gracefully stop RabbitMQ consumer threads
        self.shutdown_rabbitMQ_consumers()
        # Step 4: Close RabbitMQ connection
        self.close_rabbitMQ_connection()
        # Step 5: Close database connection
        # self.close_database_connection()