import cv2
import numpy as np
from datetime import datetime, time
import base64
import logging
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.utils.MovementDetection import log_time
from src.utils.Logger import LoggingConfig
logger = LoggingConfig().setup_logging()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))


class FrameQualityChecker:
    """
    Utility class for grid-based frame quality analysis, detecting issues such as blur, low contrast, low variance, and weak gradients.
    Splits frames into an MxM grid and checks variance, gradient, blur, and contrast.
    Handles both single frame and multiple frames in a batch.
    """

    def __init__(self):
        self.blur_threshold = float(cfg.get_value_config(Constants.FRAME_QUALITY_THRESHOLDS, Constants.BLUR_THRESHOLD))
        self.min_contrast = float(cfg.get_value_config(Constants.FRAME_QUALITY_THRESHOLDS, Constants.MIN_CONTRAST))
        self.variance_threshold = float(cfg.get_value_config(Constants.FRAME_QUALITY_THRESHOLDS, Constants.VARIANCE_THRESHOLD))
        self.gradient_threshold = float(cfg.get_value_config(Constants.FRAME_QUALITY_THRESHOLDS, Constants.GRADIENT_THRESHOLD))
        self.grid_size = int(cfg.get_value_config(Constants.FRAME_QUALITY_THRESHOLDS, Constants.GRID_SIZE))

        self.current_frame_idx = -1

    def calculate_variance(self, gray_region):
        """
        Calculate the variance of pixel intensities in a grayscale region.

        Args:
            gray_region (np.ndarray): Grayscale image region (2D array).

        Returns:
            float: Variance of pixel intensities.
        """
        return np.var(gray_region)

    def calculate_average_gradient(self, gray_region):
        """
        Calculate the average gradient magnitude in a grayscale region.

        Args:
            gray_region (np.ndarray): Grayscale image region (2D array).

        Returns:
            float: Average gradient magnitude across the region.
        """
        grad_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_magnitude)

    def check_blur_region(self, gray_region):
        """
        Check if a grayscale region is blurry using Laplacian variance.

        Args:
            gray_region (np.ndarray): Grayscale image region (2D array).

        Returns:
            tuple:
                float: Laplacian variance (focus measure).
                bool: True if region is blurry (below threshold), else False.
        """
        if gray_region.size == 0:
            return 0, True
        fm = cv2.Laplacian(gray_region, cv2.CV_64F).var()
        return fm, fm < self.blur_threshold

    def check_contrast_region(self, gray_region):
        """
        Check the contrast of a grayscale region using normalized standard deviation.

        Args:
            gray_region (np.ndarray): Grayscale image region (2D array).

        Returns:
            tuple:
                float: Normalized contrast percentage.
                bool: True if contrast is below threshold, else False.
        """
        if gray_region.size == 0:
            return 0, True
        mean_val = gray_region.mean()
        std_val = gray_region.std()
        norm_contrast = std_val / (mean_val + 1e-5) * Constants.ONE_HUNDRED
        return norm_contrast, (norm_contrast < self.min_contrast)

    def analyze_grid_cell(self, gray_region):
        """
        Analyze a single grid cell for variance, gradient, blur, and contrast.

        Args:
            gray_region (np.ndarray): Grayscale image region (2D array).

        Returns:
            tuple:
                dict: Cell metrics {variance, avg_gradient, blur_value, contrast}.
                dict: Cell issues {low_variance, low_gradient, blurry, low_contrast}.
                bool: True if the cell is considered bad, else False.
        """
        variance = self.calculate_variance(gray_region)
        low_variance = variance < self.variance_threshold
        avg_gradient = self.calculate_average_gradient(gray_region)
        low_gradient = avg_gradient < self.gradient_threshold
        blur_val, is_blurry = self.check_blur_region(gray_region)
        contrast_val, low_contrast = self.check_contrast_region(gray_region)

        cell_issues = {
            Constants.LOW_VARIANCE: low_variance,
            Constants.LOW_GRADIENT: low_gradient,
            Constants.BLURRY: is_blurry,
            Constants.LOW_CONTRAST: low_contrast
        }

        cell_metrics = {
            Constants.VARIANCE: variance,
            Constants.AVG_GRADIENT: avg_gradient,
            Constants.BLUR_VALUE: blur_val,
            Constants.CONTRAST: contrast_val
        }

        return cell_metrics, cell_issues, any(cell_issues.values())

    def analyze_frame_grid(self, frame):
        """
        Analyze an entire frame by splitting into an MxM grid and checking quality metrics.

        Args:
            frame (np.ndarray): Input BGR frame (image).

        Returns:
            tuple:
                list: Grid analysis with metrics/issues for each cell.
                dict: Frame summary with total cells, bad cells, bad area percentage, and frame status.
        """
        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size
        total_cells = self.grid_size * self.grid_size
        bad_cells = 0
        grid_analysis = []

        for i in range(self.grid_size):
            row_analysis = []
            for j in range(self.grid_size):
                start_y = i * cell_height
                end_y = min((i + 1) * cell_height, height)
                start_x = j * cell_width
                end_x = min((j + 1) * cell_width, width)
                gray_region = gray[start_y:end_y, start_x:end_x]

                metrics, issues, is_bad = self.analyze_grid_cell(gray_region)

                if i > Constants.ZERO and is_bad:  
                    bad_cells += Constants.ONE

                row_analysis.append({
                    Constants.POSITION: (i, j),
                    Constants.METRICS: metrics,
                    Constants.ISSUES: issues,
                    Constants.IS_BAD: is_bad if i > 0 else False,
                    Constants.BOUNDS: (start_x, start_y, end_x, end_y)
                })
            grid_analysis.append(row_analysis)

        analyzed_cells = total_cells - self.grid_size 
        is_bad_frame = (bad_cells > Constants.FOUR)

        frame_summary = {
            Constants.TOTAL_CELLS: analyzed_cells,
            Constants.BAD_CELLS: bad_cells,
            Constants.BAD_AREA_PERCENTAGE: bad_cells / analyzed_cells,
            Constants.IS_BAD_FRAME: is_bad_frame,
        }
        return grid_analysis, frame_summary
    


    def decode_frame(self, encoded_str):
        """
        Decode a base64-encoded frame string to an OpenCV image.

        Args:
            encoded_str (str): Base64 encoded frame.

        Returns:
            np.ndarray: Decoded OpenCV BGR image, or None if decoding fails.
        """
        try:
            frame_bytes = base64.b64decode(encoded_str)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logging.debug(f"Frame decoding failed: {e}")
            return None

    @log_time("Time taken to analyse frame quality of one batches' frame ",True)
    def analyze_batch(self, frame_data):
        """
        Analyze a batch of frames (single or multiple) for quality issues.

        Args:
            frame_data (dict): Dictionary containing frames under Constants.FRAME.
                               - Can be a single encoded frame or list of encoded frames.

        Returns:
            bool: True if any frame in the batch has bad quality, False otherwise.
        """
        decoded_frames = []
        any_bad = False

        try:
            encoded = frame_data[Constants.FRAME_METADATA][Constants.RAW_FRAME]
            if encoded is None:
                return any_bad

            # Handle single or list of frames
            if isinstance(encoded, list):
                imgs = [self.decode_frame(f) for f in encoded]
            else:
                imgs = [self.decode_frame(encoded)]

            for img in imgs:
                if img is not None:
                    img = cv2.resize(img, (1280, 720))
                    decoded_frames.append(img)

            if not decoded_frames:
                return any_bad

            # Analyze each frame
            for img in decoded_frames:
                _, frame_summary = self.analyze_frame_grid(img)
                if frame_summary[Constants.IS_BAD_FRAME]:
                    any_bad = True  
        except Exception as e:
            logging.error(f"Batch analysis failed: {e}")
        return any_bad
    
    @log_time("Time taken to improve quality of single frame during night.",True)
    def histogram_equilizer(self,frame:np.ndarray) -> np.ndarray:
        """Helps to improve frame quality in low light conditions"""

        # Convert from BGR to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Split channels
        y, cr, cb = cv2.split(ycrcb)

        # Apply CLAHE to Y channel (luminance)
        y_clahe = clahe.apply(y)
    
        # Merge channels back
        ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    
        # Convert back to BGR
        clahe_bgr = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

        return clahe_bgr


def is_night_time(timestamp_str: str) -> bool:
    """this function helps to check if given time stamp is night or not"""

    ts = datetime.fromisoformat(timestamp_str)
    
    night_start = time(19, 0, 0)  
    night_end = time(5, 0, 0)
    ts_time = ts.time()
    return ts_time >= night_start or ts_time < night_end