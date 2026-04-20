"""
Defines the VisionPipeline class, which manages and coordinates various computer vision tasks and system resources.

This class holds global variables and initialized objects for:
- System resources (RabbitMQ message queue, GPU manager, threads, configuration, database connections, video writer).
- Detection and tracking modules (detectors, camera tracker, frame tracker, motion detector).
- Synchronization and shutdown mechanisms (locks, events).
- Other runtime parameters (scaling factor, transformation matrix, client, person ID, buffer times).

Author: HCLTech
"""
import queue
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg

class VisionPipeline:
    global_rabbitmq_message_queue=queue.Queue(maxsize=int(cfg.get_env_config(Constants.MAX_QUEUE_SIZE)))
    gpu_manager=None
    rabbitmq_threads = []
    queue_name=None
    global_config=None
    h_mat=None
    scaling_factor=None
    rabbitmq_service = None
    rabbitmq_connection=None
    shutdown_event=None
    shutdown_lock=None
    cursor=None 
    db_connection=None
    camera_tracker={}
    person_id=None
    ppekit_motion_buffer_time=None