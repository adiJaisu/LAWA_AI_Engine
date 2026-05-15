import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.detectors.custom_tracker import CustomPersonTracker

from src.Exception.Exception import PredictionError, FrameProcessingException, QueueManagementException

class QueueManagement:
    """
    Queue Management detection system.
    Waiters = People currently in Waiting Area.
    Served = People currently in Service Area.
    """

    def __init__(self, detector):
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        self.logger.info("Initializing QueueManagement detection class")
        
        config_section = Constants.QUEUE_MANAGEMENT
        
        self.person_class_id = int(cfg.get_value_config(config_section, Constants.PERSON_CLASS_ID))
        self.confidence = float(cfg.get_value_config(config_section, Constants.QUEUE_MANAGEMENT_CONFIDENCE))
        self.alert_threshold = float(cfg.get_value_config(config_section, Constants.QUEUE_MANAGEMENT_ALERT_THRESHOLD))
        
        reid_params = {
            "model_name": cfg.get_value_config(config_section, Constants.REID_MODEL_NAME),
            "num_classes": int(cfg.get_value_config(config_section, Constants.REID_NUM_CLASSES)),
            "image_size": tuple(map(int, cfg.get_value_config(config_section, Constants.REID_IMAGE_SIZE).split(","))),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "device": str(detector.device)
        }
        
        tracking_params = {
            "track_thresh": float(cfg.get_value_config(config_section, Constants.TRACK_THRESH)),
            "track_buffer": int(cfg.get_value_config(config_section, Constants.TRACK_BUFFER)),
            "match_thresh": float(cfg.get_value_config(config_section, Constants.MATCH_THRESH)),
            "min_tracklet_len": int(cfg.get_value_config(config_section, Constants.MIN_TRACKLET_LEN)),
            "embedding_weight": float(cfg.get_value_config(config_section, Constants.EMBEDDING_WEIGHT)),
            "iou_weight": float(cfg.get_value_config(config_section, Constants.IOU_WEIGHT))
        }
        
        identity_params = {
            "threshold": float(cfg.get_value_config(config_section, Constants.IDENTITY_THRESHOLD)),
            "avg_threshold": float(cfg.get_value_config(config_section, Constants.IDENTITY_AVG_THRESHOLD)),
            "keep_threshold": float(cfg.get_value_config(config_section, Constants.IDENTITY_KEEP_THRESHOLD)),
            "color_threshold": float(cfg.get_value_config(config_section, Constants.IDENTITY_COLOR_THRESHOLD)),
            "reactivation_window": int(cfg.get_value_config(config_section, Constants.IDENTITY_REACTIVATION_WINDOW)),
            "reactivation_centroid_thresh": float(cfg.get_value_config(config_section, Constants.IDENTITY_REACTIVATION_CENTROID_THRESH)),
            "reactivation_color_thresh": float(cfg.get_value_config(config_section, Constants.IDENTITY_REACTIVATION_COLOR_THRESH)),
            "long_absence_window": int(cfg.get_value_config(config_section, Constants.IDENTITY_LONG_ABSENCE_WINDOW)),
            "long_centroid_thresh": float(cfg.get_value_config(config_section, Constants.IDENTITY_LONG_CENTROID_THRESH)),
            "long_color_thresh": float(cfg.get_value_config(config_section, Constants.IDENTITY_LONG_COLOR_THRESH)),
            "max_embeddings": int(cfg.get_value_config(config_section, Constants.IDENTITY_MAX_EMBEDDINGS)),
            "max_color_hists": int(cfg.get_value_config(config_section, Constants.IDENTITY_MAX_COLOR_HISTS)),
            "score_centroid_weight": float(cfg.get_value_config(config_section, Constants.SCORE_CENTROID_WEIGHT)),
            "score_avg_weight": float(cfg.get_value_config(config_section, Constants.SCORE_AVG_WEIGHT)),
            "score_color_weight": float(cfg.get_value_config(config_section, Constants.SCORE_COLOR_WEIGHT))
        }
        
        self.tracker = CustomPersonTracker(detector, tracking_params, identity_params, reid_params, self.person_class_id)
        
        self.identities_dir = Path(cfg.get_value_config(config_section, "IDENTITY_SAVE_DIR")) if cfg.obj_config.has_option(config_section, "IDENTITY_SAVE_DIR") else Path("identities_queue")
        self.alerts_dir = Path(cfg.get_value_config(config_section, "ALERTS_DIR")) if cfg.obj_config.has_option(config_section, "ALERTS_DIR") else Path("alerts_queue")
        
        self.wait_start_times = {} # pid -> start_time
        self.id_alerted = {}
        self.alert_snapshot_index = 1
        self.frame_count = 0
        self.fps = Constants.FPS
        
        os.makedirs(self.identities_dir, exist_ok=True)
        os.makedirs(self.alerts_dir, exist_ok=True)

    def point_in_polygon(self, point, polygon):
        if len(polygon) < 3:
            return False
        pts = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

    def copy_latest_identity_crop(self, identity_id: int) -> Path | None:
        person_dir = self.identities_dir / f"person_{identity_id}"
        if not person_dir.exists():
            return None
        crop_files = list(person_dir.glob("frame_*.jpg"))
        if not crop_files: crop_files = list(person_dir.glob("*.jpg"))
        if not crop_files: return None
        sorted_files = sorted(crop_files, key=lambda p: p.stat().st_mtime)
        chosen_file = sorted_files[3] if len(sorted_files) >= 4 else sorted_files[-1]
        dest_path = self.alerts_dir / f"queue_alert_{identity_id}_{self.alert_snapshot_index}{chosen_file.suffix}"
        try:
            shutil.copy2(chosen_file, dest_path)
            self.alert_snapshot_index += 1
            return dest_path
        except Exception: return None

    def detect(self, frame: np.ndarray, rois: List[List[List[int]]], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        try:
            if not rois or len(rois) < 3:
                return frame, False, {}

            # ROI 0: Inside, ROI 1: Waiting Area, ROI 2: Service Area
            waiting_poly = np.array(rois[1], np.int32)
            service_poly = np.array(rois[2], np.int32)

            cv2.polylines(frame, [waiting_poly], True, (255, 0, 0), 2)
            cv2.polylines(frame, [service_poly], True, (0, 0, 255), 2)

            self.frame_count += 1
            current_time_s = self.frame_count / self.fps
            has_new_alert = False

            try:
                boxes, ids = self.tracker.process_frame(frame, [self.person_class_id], self.confidence, device=detector.device)
            except Exception as e:
                self.logger.error(f"Tracking error in QueueManagement: {e}")
                raise PredictionError(f"Tracking error: {e}")

            try:
                current_waiters = 0
                current_served = 0
                detected_pids = set(ids)
                bboxes_list = []

                for box, pid in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    bboxes_list.append([int(x1), int(y1), int(x2), int(y2), int(pid)])
                    foot_pt = ((x1 + x2) // 2, y2)
                    
                    is_waiting = self.point_in_polygon(foot_pt, waiting_poly)
                    is_being_served = self.point_in_polygon(foot_pt, service_poly)

                    if is_waiting:
                        current_waiters += 1
                        if pid not in self.wait_start_times:
                            self.wait_start_times[pid] = current_time_s
                    elif pid in self.wait_start_times:
                        # person left waiting area (either to service or out)
                        del self.wait_start_times[pid]

                    if is_being_served:
                        current_served += 1

                    # Alert Check for long wait
                    if pid in self.wait_start_times:
                        wait_duration = current_time_s - self.wait_start_times[pid]
                        if wait_duration >= self.alert_threshold and not self.id_alerted.get(pid, False):
                            self.id_alerted[pid] = True
                            has_new_alert = True
                            self.copy_latest_identity_crop(pid)

                    color = (0, 0, 255) if self.id_alerted.get(pid, False) else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Cleanup start times for pids no longer detected
                for pid in list(self.wait_start_times.keys()):
                    if pid not in detected_pids:
                        del self.wait_start_times[pid]

                # Draw Overlay
                bx1, by1, bx2, by2 = frame.shape[1]-220, frame.shape[0]-100, frame.shape[1]-20, frame.shape[0]-20
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                cv2.putText(frame, f"Waiters : {current_waiters}", (bx1 + 15, by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Served  : {current_served}", (bx1 + 15, by1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                status_dict = {"waiters_count": current_waiters, "served_count": current_served, "bboxes": bboxes_list}
                
                # cv2.imshow("Queue Management", frame)
                # cv2.waitKey(1)
                
                return frame, has_new_alert, status_dict
            except Exception as e:
                self.logger.error(f"Error processing detections in QueueManagement: {e}")
                raise FrameProcessingException(f"Detection processing failed: {e}")
        except (PredictionError, FrameProcessingException) as e:
            raise QueueManagementException(str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error in QueueManagement: {e}")
            raise QueueManagementException(f"QueueManagement failed: {e}")
