"""
GPUdevicemanager.py

This module provides classes for managing GPU (or CPU) resources and task execution in a multi-threaded environment.
It is designed to efficiently distribute computational tasks, such as deep learning inference, across multiple GPUs or CPUs
by utilizing worker threads, task queues, and thread-safe mechanisms.

Classes:
    - GPUWorker: Manages a single worker thread assigned to a specific GPU or CPU device. Handles task execution, busy/idle status,
      and ensures thread safety using a global lock.
    - GPUManager: Oversees multiple GPUWorker instances, handles worker initialization, task assignment, status logging,
      retry logic, and graceful shutdown of all workers.

Key Features:
    - Dynamic detection and utilization of available GPUs or fallback to CPU.
    - Thread-safe task assignment and execution using Python's threading and queue modules.
    - Support for loading and assigning deep learning models to specific devices.
    - Graceful shutdown and resource cleanup.
    - Logging of worker status and error handling for robust operation.

Intended Usage:
    - Integrate into vision or AI pipelines requiring parallelized inference or processing across multiple devices.
    - Extend or adapt for other compute-intensive workloads that benefit from distributed task management.

Dependencies:
    - torch (PyTorch)
    - threading, queue, time
    - Project-specific modules: ObjectDetector, VisionPipeline, Constants, LoggingConfig, ConfigReader

Author: HCLTech

"""
import threading
import queue
import os
import time
import torch
import ast
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.utils.Utilies import GpuManagerUtils
from src.constant.global_constant import VisionPipeline

logger = LoggingConfig().setup_logging()

class GPUWorker:
    """
    GPUWorker manages a single worker thread assigned to a specific GPU.
    It processes tasks from a queue, executes them using the provided model,
    and maintains its busy/idle status. Thread-safety is ensured using a global lock.
    """

    def __init__(self, gpu_id, worker_id):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.busy = False
        self.task_queue = queue.Queue(maxsize=Constants.ONE)
        self.thread_lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        logger.debug(f"Starting GPUWorker {self.worker_id} on GPU/CPU {self.gpu_id}")
        self.thread.start()

    def run(self):
        while not VisionPipeline.shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=Constants.ONE_SEC)
            except queue.Empty:
                continue

            with self.thread_lock:
                self.busy = True
            try:
                func, validated_msg_with_frames_and_metadatas, model, secondary_model = task
                func(model, secondary_model,validated_msg_with_frames_and_metadatas)

                if VisionPipeline.shutdown_event.is_set():
                    logger.info(f"[INFO] Shutdown signal received. Releasing worker {self.worker_id}.")
                    break

            except Exception as e:
                logger.error(f"Worker {self.worker_id} on GPU/CPU {self.gpu_id} failed: {e}", exc_info=True)
            finally:
                self.task_queue.task_done()
                with self.thread_lock:
                    self.busy = False

    def assign_task(self, func, validated_msg_with_frames_and_metadatas, model=None, secondary_model=None):
        with self.thread_lock:
            if self.busy or not self.task_queue.empty():
                logger.debug(f"Worker {self.worker_id} on GPU/CPU {self.gpu_id} is busy or queue is not empty.")
                return False

            try:
                self.task_queue.put_nowait((func, validated_msg_with_frames_and_metadatas, model, secondary_model))
                logger.debug(f"Task {func.__name__} assigned to worker {self.worker_id} on GPU/CPU {self.gpu_id}")
                return True
            except queue.Full:
                logger.warning(f"Task queue full for worker {self.worker_id} on GPU/CPU {self.gpu_id}")
                return False
            except Exception as e:
                logger.error(f"Error assigning task to worker {self.worker_id} on GPU/CPU {self.gpu_id}: {e}", exc_info=True)
                return False

    def is_available(self):
        with self.thread_lock:
            return not self.busy and self.task_queue.empty()


class GPUManager:
    """
    GPUManager manages multiple GPUWorker instances across available GPUs.
    It handles worker initialization, task assignment, status logging, retry logic, and graceful shutdown.
    """

    def __init__(self, num_workers_per_gpu: int, model_path:str=None, secondary_model_path:str=None):
        self.device = cfg.get_env_config(Constants.DEVICE)
        self.workers = []

        try:            
            if self.device == Constants.CUDA:
                available_gpus = os.environ.get(Constants.AVAILABLE_GPU_INDEX)
                if available_gpus :
                    if  -1 not in ast.literal_eval(available_gpus):
                        self.gpus = ast.literal_eval(available_gpus)
                        logger.info(F"Found GPUs indexs from environment {available_gpus}")
                else:
                    self.gpus = list(range(torch.cuda.device_count()))
                logger.debug(f"Available GPUs: {self.gpus}")

                for gpu_id in self.gpus:
                    for n in range(num_workers_per_gpu):
                        primary_model,secondary_model = GpuManagerUtils.get_primary_and_secondary_model(primary_model_path=model_path,secondary_model_path=secondary_model_path,device=gpu_id)
                        worker_id = f"{gpu_id}_{n}"
                        worker = GPUWorker(gpu_id, worker_id)
                        self.workers.append({
                            Constants.WORKER: worker,
                            Constants.MODEL:  primary_model,
                            Constants.SECONDARY_MODEL: secondary_model
                        })
                        logger.debug(f"Initialized worker {n} on GPU {gpu_id}")
                        time.sleep(Constants.THREE_SEC)
            else:
                for n in range(num_workers_per_gpu):
                    primary_model,secondary_model = GpuManagerUtils.get_primary_and_secondary_model(primary_model_path=model_path,secondary_model_path=secondary_model_path,device=self.device)
                    worker_id = f"{self.device}_{n}"
                    worker = GPUWorker(self.device, worker_id)
                    self.workers.append({
                            Constants.WORKER: worker,
                            Constants.MODEL: primary_model,
                            Constants.SECONDARY_MODEL: secondary_model
                        })
                    logger.debug(f"Initialized worker {n} on CPU {self.device}")
        except Exception as e:
            logger.error(f"Error initializing GPUManager: {e}", exc_info=True)
            raise

    def assign_task(self, func, validated_msg_with_frames_and_metadatas):
        try:
            for worker_data in self.workers:
                worker:GPUWorker = worker_data[Constants.WORKER]
                if worker.is_available():
                    if worker.assign_task(func,validated_msg_with_frames_and_metadatas,model=worker_data[Constants.MODEL],secondary_model=worker_data.get(Constants.SECONDARY_MODEL)):
                        logger.debug(f"Assigned task to worker {worker.worker_id} on GPU/CPU {worker.gpu_id}")
                        return True
            logger.warning("No available workers currently.")
            return False
        except Exception as e:
            logger.error(f"Error assigning task in GPUManager: {e}", exc_info=True)
            return False


    def log_worker_status(self):
        """
        Log the status (busy/idle) of all workers.
        """
        logger.debug("Logging status of all GPU workers.")
        try:
            for worker_data in self.workers:
                worker = worker_data[Constants.WORKER]
                status = Constants.BUSY if worker.busy else Constants.IDLE
                logger.debug(f"Worker {worker.worker_id} on {worker.gpu_id} - {status}")
        except Exception as e:
            logger.error(f"Error logging worker status: {e}", exc_info=True)

    def retry_assign_task(self, func, *args, delay=0.01, **kwargs):
        """
        Wait until a worker becomes available and assign the task.
        Retries indefinitely until successful.

        Args:
            func (callable): The function to execute.
            *args: Arguments for the function.
            delay (float): Delay in seconds between retries.
            **kwargs: Keyword arguments for the function.

        Returns:
            bool: True when the task is successfully assigned.
        """
        attempt = Constants.ZERO
        logger.debug("Retrying task assignment until a worker is available.")
        while True:
            try:
                if self.assign_task(func, *args, **kwargs):
                    logger.debug(f"Task assigned after {attempt} attempts.")
                    return True
                attempt += 1
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error during retry_assign_task: {e}", exc_info=True)
                time.sleep(delay)

    def shutdown(self):
        """
        Gracefully shutdown all workers and wait for their threads to finish.
        """
        logger.info("Shutting down GPU manager...")
        try:
            VisionPipeline.shutdown_event.set()
            # Wait for all worker threads to finish
            for worker_data in self.workers:
                worker = worker_data[Constants.WORKER]
                logger.info(f"Waiting for worker {worker.worker_id} on GPU {worker.gpu_id} to terminate.")
                worker.thread.join(timeout=Constants.FIVE_SEC)
            logger.info("[INFO] GPU manager shutdown complete.")
        except Exception as e:
            logger.error(f"Error during GPUManager shutdown: {e}", exc_info=True)