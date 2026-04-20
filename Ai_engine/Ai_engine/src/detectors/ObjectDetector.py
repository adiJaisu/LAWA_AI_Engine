import torch
import numpy as np
from ultralytics import YOLO
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.Exception.Exception import (ModelLoadingError, PredictionError)

class ObjectDetector:
    """
    This Class is used to detect objects in a frame using YOLO.
    """

    def __init__(self, model_path: str = None, device = None):
        """
        Initialize the ObjectDetector instance.

        Args:
            model_path (str, optional): Path to the YOLO model file. If None, model is not loaded.
            device (str, optional): Device to load the model on ('cuda' or 'cpu'). If None, auto-detects.

        Attributes:
            model_path (str): Path to the YOLO model.
            device (str): Device used for inference.
            logger (logging.Logger): Logger instance for logging messages.
            model (YOLO, optional): Loaded YOLO model instance.
        """
        self.model_path = model_path

        # Logging setup
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        # Handle int device implicitly passed from GPUManager or string digit config
        if isinstance(device, str) and device.isdigit():
            device = int(device)
            
        if isinstance(device, int) and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device}')
        elif isinstance(device, str) and device != 'cuda' and device != 'cpu' and not device.startswith('cuda:'):
            # Safe fallback if user explicitly tries to pass something like '0' natively to pyTorch which crashes
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
            
        self.logger.info(f"Model will be loaded on device: {self.device}")
        
        if self.model_path is None:
            self.logger.warning("Model path not provided. ObjectDetector initialized without loading a model.")
            self.model = None
        else:
            self.model = self._load_model()
            
    def _load_model(self) -> YOLO:
        """
        Loads the YOLO model from the specified path onto the selected device.

        Returns:
            YOLO: The loaded YOLO model instance.

        Raises:
            ModelLoadingError: If the model fails to load.
        """
        try:
            self.logger.info(f"Loading YOLO model from: {self.model_path}")
            model = YOLO(self.model_path)
            model = model.to(self.device)
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise ModelLoadingError(f"Failed to load model from {self.model_path}") from e

    def make_prediction(self, frame: any, classes_id: list[int], confidence: float, img_size: int = 640) -> list | None:
        """
        Perform object detection prediction on a given frame using the loaded model.
        Args:
            frame (any): The input image/frame to perform prediction on.
            classes_id (list[int]): List of class IDs to detect.
            confidence (float): Confidence threshold for predictions (between 0.0 and 1.0).
            img_size (int, optional): Size to which the input image will be resized. Defaults to 640.
        Returns:
            list | None: The prediction results as a list if successful, otherwise None.
        Raises:
            ValueError: If the confidence value is not between 0.0 and 1.0.
            PredictionError: If the model or its attributes are not properly initialized.
        """
        try:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("Confidence should be between 0.0 and 1.0")

            if not isinstance(img_size,tuple) and img_size>640:
                self.logger.warning(f"Image size {img_size} is larger than 640, resizing to 640.")
                img_size = 640
            results = self.model.predict(frame, classes=classes_id, conf=confidence,device=self.device,imgsz=img_size)

            if not results or not isinstance(results, list) or len(results) == Constants.ZERO:
                self.logger.error("Prediction failed: results are empty or invalid.")
                return None

            if not hasattr(results[Constants.ZERO], Constants.BOXES):
                self.logger.error("Prediction failed: 'boxes' attribute not found in results.")
                return None

            return results
        
        except PredictionError as e:
            self.logger.error(f"Model or attributes not properly initialized: {e}")

    def make_prediction_with_tracking(self, frame: np.ndarray, classes_id: list[int], confidence: float, inference_image_size: int = 640, tracker: str = "bytetrack.yaml", iou: float = Constants.ZERO_POINT_FIVE) -> list:
        """
        Perform object detection with tracking on a given frame using the loaded model.

        Args:
            frame (any): The input image/frame to perform prediction and tracking on.
            classes_id (list[int]): List of class IDs to detect and track.
            confidence (float): Confidence threshold for predictions (between 0.0 and 1.0).
            inference_image_size (int): Size to which the input image will be resized.
            tracker (str): String identifier for the tracker yaml config.

        Returns:
            list: The tracking prediction results as a list if successful, otherwise an empty list.
        Raises:
            Exception: If model tracking fails.
        """
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker,
                classes=classes_id,
                verbose=True,
                conf=confidence, 
                imgsz=inference_image_size,
                iou= iou,
                device= self.device
            )
            return results
        except Exception as e:
            self.logger.error(f"Model tracking failed: {e}")
            return []