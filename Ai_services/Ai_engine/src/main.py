import os
import signal
import time
import queue
import threading
import numpy as np
from typing import List, Dict, Any
from src.constant.constants import Constants
from src.constant.global_constant import VisionPipeline
from src.utils.Utilies import Utility, ResourcesCleaner
from concurrent.futures import ThreadPoolExecutor
from src.utils.Streamhandler_validation.Streamhandler_validation import Validator
from src.utils.Logger import LoggingConfig, log_time
from src.utils.ResultProcessing import process_detection_results
from src.utils.ResultProcessing import send_msg_to_event_manager

threading_pool_executer = ThreadPoolExecutor(Constants.TEN)
VisionPipeline.queue_name = os.environ.get(Constants.QUEUE_NAME)
VisionPipeline.shutdown_event = threading.Event()
VisionPipeline.shutdown_lock = threading.Lock()
signal.signal(signal.SIGTERM, Utility.handle_shutdown_signal)
signal.signal(signal.SIGINT, Utility.handle_shutdown_signal)
logger = LoggingConfig().setup_logging()
validator = Validator()
cleaner = ResourcesCleaner()

@log_time("Time taken to complete one batch of processing", True)
def dispatch_vision_task(detector: Any, secondary_model: Any, validated_msg_with_frames_and_metadatas: List[Dict]) -> None:
    """Dispatches and executes the appropriate vision detection task for a given use case."""
    
    usecase_name = validated_msg_with_frames_and_metadatas[Constants.ZERO][Constants.FRAME_METADATA][Constants.USECASE_NAME]
    try:
        if usecase_name == Constants.LOITERING_DETECTION_USECASE:
            from src.executers.LoiteringDetector_executer import execute_loitering_detection
            # In loitering detection we just use the primary detector for tracking
            results = execute_loitering_detection(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.TAILGATE_DETECTION_USECASE:
            from src.executers.TailgateDetector_executer import execute_tailgate_detection
            results = execute_tailgate_detection(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.IN_OUT_PERSON_COUNT_USECASE:
            from src.executers.InOutPersonCount_executer import execute_in_out_person_count
            results = execute_in_out_person_count(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.TRAIN_ARRIVAL_DEPART_MONITOR_USECASE:
            from src.executers.TrainArrivalDepartMonitor_executer import execute_train_arrival_depart_monitor
            results = execute_train_arrival_depart_monitor(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.CROWD_DENSITY_USECASE:
            from src.executers.CrowdDensity_executer import execute_crowd_density
            results = execute_crowd_density(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.PERSON_ENTERED_INSIDE_TRAIN_USECASE:
            from src.executers.PersonEnteredInsideTrain_executer import execute_person_entered_inside_train
            results = execute_person_entered_inside_train(validated_msg_with_frames_and_metadatas, detector=detector)
        elif usecase_name == Constants.PERSON_COUNT_INSIDE_COMPARTMENT_USECASE:
            from src.executers.PersonCountInsideCompartment_executer import execute_person_count_inside_compartment
            results = execute_person_count_inside_compartment(validated_msg_with_frames_and_metadatas, detector=detector)
        else:
            logger.error(f"dispatch_vision_task: Unsupported use case: {usecase_name}")
            return

        event_manager_evidence = process_detection_results(results)
        if event_manager_evidence and len(event_manager_evidence) != Constants.ZERO:
            logger.debug(f"total evidence for event_manager of length {len(event_manager_evidence)}")
            threading_pool_executer.submit(send_msg_to_event_manager, event_manager_evidence)
        
    except Exception as e:
        logger.error(f"dispatch_vision_task: Error processing use case {usecase_name}: {e}", exc_info=True)


async def process_vision_stream() -> None:
    """Main asynchronous loop for processing the vision stream pipeline."""

    try:        
        while not VisionPipeline.shutdown_event.is_set():
            try:
                # Get a batch of messages from the queue
                messages = VisionPipeline.global_rabbitmq_message_queue.get(timeout=1)
                logger.debug(f"[VISION STREAM] remaining messages in queue: {VisionPipeline.global_rabbitmq_message_queue.qsize()}")
            except queue.Empty:
                time.sleep(Constants.ZERO_POINT_ONE)
                logger.debug("[VISION STREAM] No messages in queue, waiting for new messages...")
                continue  

            validated_msg_with_frames_and_metadatas = Utility.preprocess_messages(messages=messages, validator=validator)

            if validated_msg_with_frames_and_metadatas is not None:
                # Assign task to the GPU manager
                task_assigned = VisionPipeline.gpu_manager.assign_task(dispatch_vision_task, validated_msg_with_frames_and_metadatas)
                logger.info(f"Task assigned: {task_assigned}")
                
                if not task_assigned and not VisionPipeline.shutdown_event.is_set():
                    logger.debug(f"[VISION STREAM] Failed to assign task. Retrying...")
                    VisionPipeline.gpu_manager.retry_assign_task(dispatch_vision_task, validated_msg_with_frames_and_metadatas)
            else:
                logger.debug(f"No valid Message found in this batch")
            
            VisionPipeline.gpu_manager.log_worker_status()
            if VisionPipeline.shutdown_event.is_set():
                logger.info("[VISION STREAM] Shutdown signal received. Exiting main loop.")
                break   
                        
    except Exception as e:
        logger.critical(f"[VISION STREAM] Critical error in main process: {str(e)}", exc_info=True)
    finally:
        logger.info("[VISION STREAM] Received shutdown signal. Performing final cleanup...")
        await cleaner.cleanup_resources()