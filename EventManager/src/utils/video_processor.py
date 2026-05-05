import cv2
import os
import base64
import numpy as np
import datetime
import subprocess
import tempfile
from pathlib import Path
from src.utils.logger import LoggingConfig
from src.utils.ConfigReader import cfg
from src.constant.constants import Constants
from src.exception.exception import VideoProcessingError

logger = LoggingConfig.setup_logging()

class VideoEvidenceProcessor:
    def __init__(self, base_storage_path="./evidence_storage"):
        """Initializes the processor and ensures the base storage directory exists."""
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Read FPS from config, defaulting to 1
        fps_config = cfg.get_env_config(Constants.EVIDENCE_VIDEO_FPS)
        self.fps = str(fps_config) if fps_config else "1"

    def _decode_frame(self, base64_string):
        """Decodes base64 string back to a NumPy image."""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise VideoProcessingError(f"Failed to decode base64 frame: {e}")

    def process_frames(self, camera_id, usecase_name, frames_base64):
        """
        Creates a video immediately from the provided batch of frames.
        
        Args:
            camera_id (str): Unique ID of the camera.
            usecase_name (str): Name of the detected usecase.
            frames_base64 (list): List of base64 encoded JPEG strings.
            
        Returns:
            str: Absolute path to the saved video file, else None.
        """
        if not frames_base64:
            raise VideoProcessingError(f"[{camera_id}] No frames provided to process.")

        video_frames = frames_base64

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.base_path / camera_id / usecase_name.replace(" ", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{camera_id}_{timestamp}.mp4"
        video_path = str(save_dir / filename)

        # Create a temporary directory to hold the frames
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_frames_count = 0
            for i, b64_frame in enumerate(video_frames):
                frame = self._decode_frame(b64_frame)
                if frame is not None:
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    valid_frames_count += 1
            
            if valid_frames_count == 0:
                raise VideoProcessingError(f"[{camera_id}] Failed to decode all frames. No valid frames to encode.")

            # Use ffmpeg to create the video from the images
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", self.fps,
                "-i", os.path.join(temp_dir, "frame_%04d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                logger.error(f"[{camera_id}] ffmpeg failed: {e.stderr.decode()}")
                raise VideoProcessingError(f"ffmpeg subprocess failed: {e.stderr.decode()}")

        logger.info(f"[{camera_id}] Batch Evidence video saved at {self.fps} FPS: {video_path}")
        return os.path.abspath(video_path)
