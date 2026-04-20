"""
RabbitmqConsumer.py

This module provides the RabbitMQConsumerManager class, which manages the consumption of messages from a RabbitMQ queue using multiple threads. Each thread establishes its own connection to RabbitMQ, consumes messages in batches, and pushes them to a shared global queue for further processing. The consumer threads support graceful shutdown via a threading event.

Classes:
    RabbitMQConsumerManager: Manages multiple threaded RabbitMQ consumers for a specified queue.

Usage:
    - Initialize RabbitMQConsumerManager with the target queue name and desired number of consumer threads.
    - Call start_multiple_consumer_threads() to begin consuming messages in parallel.
    - Threads and message queues are managed globally for coordinated shutdown and message handling.

Dependencies:
    - src.constant.global_constant.VisionPipeline: Provides global shutdown event and message queue.
    - src.utils.RabbitmqServices.RabbitmqServices.initialize_rabbitmq_connection: Initializes RabbitMQ connections.
    - src.utils.Logger.LoggingConfig: Configures logging for consumer threads.
    - threading: Used for managing consumer threads.

Note:
    Ensure that the global VisionPipeline object is properly initialized before starting consumers.
    
Author: HCLTech
"""

import time,os
import threading
from src.constant.global_constant import VisionPipeline
from src.constant.constants import Constants
from src.utils.RabbitmqServices.RabbitmqServices import initialize_rabbitmq_connection
from src.Exception.Exception import RabbitMQConnectionError
import socket
from src.utils.Logger import LoggingConfig


class RabbitMQConsumerManager:
    def __init__(self, queue_name: str,num_threads: int = 4):
        self.queue_name = queue_name
        self.num_threads = num_threads

        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()

    def start_consumer_process(self, queue_name):
        """
        RabbitMQ consumer logic for each thread: connects, consumes, and pushes to shared queue.
        Implements a robust infinite loop to handle connection failures and ensuring threads stay alive until shutdown.
        """
        self.logger.info(f"[Consumer Thread] Starting consumer loop for queue: {queue_name}")
        
        while not VisionPipeline.shutdown_event.is_set():
            try:
                rabbitmq_service = initialize_rabbitmq_connection()
                rabbitmq_service.receive_messages_batch(queue_name=queue_name)
            except (RabbitMQConnectionError, socket.gaierror) as e:
                self.logger.warning(f"[Consumer Thread] Connection failed (retrying in loop)")
                if Constants.RABBITMQ_SERVICE in locals():
                    rabbitmq_service.close_connection()
            except Exception as e:
                self.logger.error(f"[Consumer Thread] Exception in consumer loop")
                if Constants.RABBITMQ_SERVICE in locals():
                    rabbitmq_service.close_connection()
            
            if not VisionPipeline.shutdown_event.is_set():
                self.logger.info(f"[Consumer Thread] Consumer Process stopped unexpectedly. Restarting in {Constants.FIVE_SEC} seconds...")
                VisionPipeline.shutdown_event.wait(Constants.FIVE_SEC)

        self.logger.info(f"[Consumer Thread] Exiting consumer process for {queue_name}")

    def _start_single_thread(self, index: int, queue_name:str) -> threading.Thread:
        """
        Start a single consumer thread and return it.
        """
        thread = threading.Thread(
            target=self.start_consumer_process,
            args=(queue_name,),
            name=f"RabbitMQConsumerThreadAI-{Constants.USECASE_QUEUE_MAPPING[queue_name]}-{index}",
            daemon=True
        )
        thread.start()
        self.logger.info(f"Started thread {thread.name}")
        return thread

    def start_multiple_consumer_threads(self):
        """
        Start multiple threads for RabbitMQ consumers. Each thread gets its own connection.
        Threads are stored globally for graceful shutdown later.
        """
        self.logger.info(f"Starting {self.num_threads} RabbitMQ consumer threads for queue '{self.queue_name}'")
        for i in range(self.num_threads):
            thread = self._start_single_thread(i,self.queue_name)
            VisionPipeline.rabbitmq_threads.append(thread)
            
        # creating a one extra thread to monitor the health of consumer threads and restart if any thread is dead.
        # This thread will also be stopped gracefully during shutdown.

        consumers_health_check_thread = threading.Thread(
            target=self.check_and_restart_threads,
            name=f"health check thread",
            daemon=True
        )
        consumers_health_check_thread.start()
        VisionPipeline.rabbitmq_threads.append(consumers_health_check_thread)

    def check_and_restart_threads(self):
        """
        Continuously check and restart dead threads until shutdown is requested.
        """
        while not VisionPipeline.shutdown_event.is_set():
            self.logger.debug("Checking consumer threads health...")
            for i, thread in enumerate(VisionPipeline.rabbitmq_threads):
                if thread.name != "health check thread":
                    if not thread.is_alive():
                        self.logger.warning(f"Thread {thread.name} is dead. Restarting...")
                        new_thread = self._start_single_thread(i,queue_name=os.environ.get(Constants.QUEUE_NAME))
                        VisionPipeline.rabbitmq_threads[i] = new_thread
                    else:
                        self.logger.debug(f"Thread {thread.name} is alive")
            time.sleep(Constants.TEN_SEC)
