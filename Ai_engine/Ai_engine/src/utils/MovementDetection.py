"""
This module provides the `detect_motion` function, designed to identify motion
within specified Regions of Interest (ROIs) between consecutive video frames.

The function performs the following operations:
- **Frame Preprocessing**: Converts input frames to grayscale and applies Gaussian blurring to reduce noise.
- **ROI-based Analysis**: Iterates through a list of polygonal ROIs, creating a mask for each.
- **Motion Calculation**: Computes the absolute difference between the masked previous and current frames,
  applies a binary threshold, and calculates the percentage of motion (white pixels) within each ROI.
- **Motion Detection**: Determines if significant motion is present in any ROI based on a configurable threshold.
- **Debugging Visualization (Optional)**: Can generate a debug frame with ROI outlines and motion percentages for visual inspection.
- **Error Handling**: Catches and logs exceptions, raising a `MotionDetectionException` for robust failure management.

Author: HCLTech
"""

import cv2
import time
import numpy as np
from src.Exception.Exception import (FrameProcessingException,MotionDetectionException)
from src.constant.constants import Constants
from src.utils.Logger import LoggingConfig,log_time
logger = LoggingConfig().setup_logging()
import cv2
import numpy as np

@log_time("Time taken to run motion detection on single frame",True)
def detect_motion(prev_frame, current_frame, roi_polygons, debug=False,camera_height = Constants.TWENTY_FIVE):
    """
    Detect motion between two frames within specified regions of interest.
    
    Args:
        prev_frame (numpy.ndarray): Previous frame (original BGR)
        current_frame (numpy.ndarray): Current frame (original BGR)
        roi_polygons (list): List of numpy arrays representing polygon ROIs
        debug (bool): Whether to create and return debug visualization
        
    Returns:
        bool: True if motion detected in any ROI
    """
    try:
        # Input validation
        if prev_frame is None or current_frame is None:
            raise ValueError("Input frames cannot be None")
        
        if not isinstance(roi_polygons, list) or not roi_polygons:
            raise ValueError("roi_polygons must be a non-empty list")

        # Convert frames to grayscale and blur
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # prev_frame = cv2.GaussianBlur(prev_frame, (Constants.SEVEN, Constants.SEVEN), Constants.ZERO)

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # gray_current = cv2.GaussianBlur(gray_current, (Constants.SEVEN, Constants.SEVEN), Constants.ZERO)
        
        motion_detected = False
        motion_data = []
        debug_frame = current_frame.copy() if debug else None
        
        for idx, roi_poly in enumerate(roi_polygons):
            # Convert ROI points to numpy array and reshape for OpenCV
            roi_points = np.array(roi_poly, dtype=np.int32).reshape((Constants.MINUS_ONE, Constants.ONE, Constants.TWO))
            
            # Create mask for this ROI
            mask = np.zeros_like(prev_frame, dtype=np.uint8)
            cv2.fillPoly(mask, [roi_points], Constants.TWO_FIVE_FIVE)
            
            # Apply mask to both frames
            prev_roi = cv2.bitwise_and(prev_frame, prev_frame, mask=mask)
            curr_roi = cv2.bitwise_and(gray_current, gray_current, mask=mask)
            
            # Calculate frame difference
            frame_delta = cv2.absdiff(prev_roi, curr_roi)
            _, thresh = cv2.threshold(frame_delta, Constants.FIFTEEN, Constants.TWO_FIVE_FIVE, cv2.THRESH_BINARY)
            
            # Calculate motion percentage
            white_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            roi_area = cv2.contourArea(roi_points)
            white_pixel_percentage = (white_pixels / roi_area) * Constants.ONE_HUNDRED if roi_area > Constants.ZERO else Constants.ZERO
            
            # Determine if motion is detected in this ROI
            if camera_height<=Constants.TWENTY:
                roi_motion_detected = white_pixel_percentage > Constants.TWO_POINT_FIVE  # Threshold as needed
            else:
                roi_motion_detected = white_pixel_percentage > Constants.ZERO_POINT_TWO_FIVE
                
            motion_detected |= roi_motion_detected
            
            if debug:
                x, y, w, h = cv2.boundingRect(roi_points)
                # Draw ROI outline
                cv2.polylines(debug_frame, [roi_points], isClosed=True, 
                            color=(255, 0, 0), thickness=Constants.TWO)
                # Draw white pixel percentage
                cv2.putText(debug_frame, f"Motion: {white_pixel_percentage:.2f}%", 
                          (x, y - Constants.TEN), cv2.FONT_HERSHEY_SIMPLEX, 
                          Constants.ZERO_POINT_FIVE, (0, 0, 255) if roi_motion_detected else (255, 255, 255), Constants.TWO)
        
        return motion_detected

    except Exception as e:
        logger.error(f"Motion detection failed: {str(e)}")
        raise MotionDetectionException(f"Motion detection failed: {str(e)}")
    

def is_frame_stuck(prev_frame, current_frame, debug=False):
    """
    Detect if the frame is 'stuck' (i.e., no significant motion) between two frames.
    This checks the entire frame, not just ROIs.

    Args:
        prev_frame (numpy.ndarray): Previous frame (original BGR)
        current_frame (numpy.ndarray): Current frame (original BGR)
        debug (bool): Whether to create and return debug visualization

    Returns:
        bool: True if frame is stuck (no significant motion)
    """
    try:
        # Input validation
        if prev_frame is None or current_frame is None:
            raise ValueError("Input frames cannot be None")

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        frame_delta = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_delta, Constants.TEN, Constants.TWO_FIVE_FIVE, cv2.THRESH_BINARY)
        kernal = np.ones((Constants.THREE, Constants.THREE), np.uint8)
        errosion = cv2.erode(thresh, kernal, iterations=Constants.ONE)

        # Calculate motion percentage
        white_pixels = cv2.countNonZero(errosion)
        frame_stuck = white_pixels == 0

        if debug:
            debug_frame = current_frame.copy()
            cv2.putText(debug_frame, f"Motion: {white_pixels}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        Constants.ZERO_POINT_FIVE, (0, 0, 255) if not frame_stuck else (255, 255, 255), Constants.TWO)
            return frame_stuck, debug_frame

        return frame_stuck

    except Exception as e:
        logger.error(f"Frame stuck detection failed: {str(e)}")
        raise MotionDetectionException(f"Frame stuck detection failed: {str(e)}")