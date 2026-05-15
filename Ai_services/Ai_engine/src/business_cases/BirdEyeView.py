import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.Logger import LoggingConfig
from src.constant.constants import Constants
from src.utils.ConfigReader import cfg
from src.Exception.Exception import PredictionError, FrameProcessingException, BirdEyeViewException

class BirdEyeView:
    """
    Bird Eye View Fusion and Tracking system.
    Fuses detections from multiple cameras into a single top-down view using Parallel Batch Inference.
    """

    def __init__(self, detector):
        logging_config = LoggingConfig()
        self.logger = logging_config.setup_logging()
        self.logger.info("Initializing BirdEyeView class with Parallel Batch Inference support")
        
        config_section = Constants.BIRD_EYE_VIEW
        
        self.confidence = float(cfg.get_value_config(config_section, Constants.BIRD_EYE_VIEW_CONFIDENCE))
        self.real_width = int(cfg.get_value_config(config_section, "REAL_WIDTH_METERS"))
        self.real_height = int(cfg.get_value_config(config_section, "REAL_HEIGHT_METERS"))
        self.pixels_per_meter = int(cfg.get_value_config(config_section, "PIXELS_PER_METER"))
        self.person_class_id = int(cfg.get_value_config(config_section, Constants.PERSON_CLASS_ID))
        self.duplicate_distance = int(cfg.get_value_config(config_section, "DUPLICATE_DISTANCE"))
        self.dist_threshold_pixels = float(cfg.get_value_config(config_section, "DIST_THRESHOLD_METERS")) * self.pixels_per_meter
        self.max_missing = int(cfg.get_value_config(config_section, "MAX_MISSING"))
        self.expected_cameras = int(cfg.get_value_config(config_section, "EXPECTED_CAMERAS"))

        self.bev_width = self.real_width * self.pixels_per_meter
        self.bev_height = self.real_height * self.pixels_per_meter

        # Tracker state
        self.next_id = 0
        self.tracks = {} # tid -> {"pos": (x, y), "missing": 0}
        
        self.frame_count = 0

    def detect_heads_batch(self, frames: List[np.ndarray], detector: Any) -> List[List[Dict]]:
        """
        Detect persons in multiple frames in parallel (Batch Inference).
        """
        try:
            # Parallel Batch Inference call to the GPU
            results_list = detector.make_prediction_with_tracking(frames, classes_id=[self.person_class_id], confidence=self.confidence)
        except Exception as e:
            self.logger.error(f"Batch Prediction Error in BEV: {e}")
            raise PredictionError(f"BEV Batch Inference failed: {e}")

        try:
            batch_detections = []
            
            # Process results for each frame in the batch
            for results in results_list:
                detections = []
                if results and hasattr(results, 'boxes') and results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (y2 - y1) < 20: # Filter small/noisy boxes
                            continue
                        
                        # Using center/foot point for BEV mapping
                        foot_x = (x1 + x2) // 2
                        foot_y = (y1 + y2) // 2
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "point": (foot_x, foot_y)
                        })
                
                # Remove close duplicates in the same frame
                filtered = []
                for det in detections:
                    fx, fy = det["point"]
                    keep = True
                    for f in filtered:
                        fx2, fy2 = f["point"]
                        dist = np.linalg.norm(np.array([fx, fy]) - np.array([fx2, fy2]))
                        if dist < self.duplicate_distance:
                            keep = False
                            break
                    if keep:
                        filtered.append(det)
                
                batch_detections.append(filtered)
                
            return batch_detections
        except Exception as e:
            self.logger.error(f"Error processing batch detection results: {e}")
            raise FrameProcessingException(f"BEV post-detection logic failed: {e}")

    def process_fused_frames(self, batch_messages: List[Dict], detector: Any) -> Tuple[np.ndarray, bool, Dict]:
        """
        Process a synchronized batch of frames from multiple cameras using Parallel Batch Inference.
        """
        try:
            all_detections = []
            annotated_frames = {}

            # The ROIs are expected to be the same for all messages in the batch
            rois = batch_messages[0][Constants.FRAME_METADATA][Constants.ROIS]
            N = self.expected_cameras
            dst_points = np.float32(rois[2 * N])

            # 1. Extract frames for Batch Inference
            frames_to_process = [msg[Constants.FRAME_METADATA][Constants.FRAME] for msg in batch_messages]
            
            # 2. Run Parallel Detection on GPU (Uses PredictionError internally)
            all_cam_dets = self.detect_heads_batch(frames_to_process, detector)

            # 3. Coordinate Mapping and Point Fusion
            for idx, msg in enumerate(batch_messages):
                frame = msg[Constants.FRAME_METADATA][Constants.FRAME]
                camera_id = msg[Constants.CAMERA_METADATA][Constants.CAMERA_ID]
                
                # Identify camera index
                try:
                    cam_idx = int(camera_id.split("_")[-1]) - 1
                except:
                    cam_idx = 0
                
                if cam_idx >= self.expected_cameras:
                    continue
                
                cam_poly = np.float32(rois[cam_idx])
                src_points = np.float32(rois[N + cam_idx])
                
                # Transform to BEV plane
                H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
                
                # Get detections for this specific camera from batch results
                cam_dets = all_cam_dets[idx]
                
                for det in cam_dets:
                    px, py = det["point"]
                    
                    # Boundary check
                    if cv2.pointPolygonTest(cam_poly, (float(px), float(py)), False) < 0:
                        continue
                    
                    # Apply Perspective Transform
                    point = np.array([[[px, py]]], dtype=np.float32)
                    bev_point = cv2.perspectiveTransform(point, H)
                    bev_x = int(bev_point[0][0][0])
                    bev_y = int(bev_point[0][0][1])
                    
                    all_detections.append({
                        "bbox": det["bbox"],
                        "bev": (bev_x, bev_y),
                        "cam_idx": cam_idx
                    })

                # Visual annotations
                cv2.polylines(frame, [cam_poly.astype(int)], True, (0, 255, 255), 3)
                annotated_frames[cam_idx] = frame

            # 4. Global Tracking across all cameras
            self.update_tracks(all_detections)

            # 5. Visualization Generation
            bev_map = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
            active_count = 0

            for tid, tr in self.tracks.items():
                if tr["missing"] > 0:
                    continue
                
                x, y = tr["pos"]
                if 0 <= x < self.bev_width and 0 <= y < self.bev_height:
                    cv2.circle(bev_map, (x, y), 6, (0, 255, 0), -1)
                    x_m, y_m = x / self.pixels_per_meter, y / self.pixels_per_meter
                    cv2.putText(bev_map, f"ID {tid} ({x_m:.1f}m,{y_m:.1f}m)", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    active_count += 1

            cv2.putText(bev_map, f"Total Count: {active_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Draw IDs back on original camera frames
            for det in all_detections:
                if "id" in det:
                    tid = det["id"]
                    cam_idx = det["cam_idx"]
                    x1, y1, x2, y2 = det["bbox"]
                    frame = annotated_frames[cam_idx]
                    color = (0, 255, 0) if cam_idx == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Dashboard assembly
            h, w = 540, 960
            vis_cam0 = cv2.resize(annotated_frames.get(0, np.zeros((h, w, 3), dtype=np.uint8)), (w, h))
            vis_cam1 = cv2.resize(annotated_frames.get(1, np.zeros((h, w, 3), dtype=np.uint8)), (w, h))
            cams_stack = np.vstack((vis_cam0, vis_cam1))
            
            bev_vis = cv2.resize(bev_map, (int(bev_map.shape[1] * (cams_stack.shape[0] / bev_map.shape[0])), cams_stack.shape[0]))
            final_vis = np.hstack((cams_stack, bev_vis))

            status_dict = {
                "total_count": active_count,
                "bboxes": [det.get("bbox") for det in all_detections]
            }
            
            # cv2.imshow("Bird Eye View Parallel Dashboard", final_vis)
            # cv2.waitKey(1)

            return final_vis, False, status_dict

        except Exception as e:
            # Re-raise as the specific BirdEyeViewException
            self.logger.error(f"BEV Parallel Fusion Error: {e}")
            raise BirdEyeViewException(f"BEV fusion logic failed: {e}")

    def update_tracks(self, detections: List[Dict]):
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {"pos": det["bev"], "missing": 0}
                det["id"] = self.next_id
                self.next_id += 1
            return

        track_ids = list(self.tracks.keys())
        track_positions = [self.tracks[t]["pos"] for t in track_ids]
        det_positions = [d["bev"] for d in detections]

        if len(det_positions) == 0:
            for tid in self.tracks:
                self.tracks[tid]["missing"] += 1
        else:
            D = np.zeros((len(det_positions), len(track_positions)))
            for i, dpos in enumerate(det_positions):
                for j, tpos in enumerate(track_positions):
                    D[i, j] = np.linalg.norm(np.array(dpos) - np.array(tpos))

            matched_dets = set()
            matched_tracks = set()

            while True:
                if D.size == 0: break
                min_val = np.min(D)
                if min_val > self.dist_threshold_pixels or min_val == np.inf:
                    break

                i, j = np.unravel_index(np.argmin(D), D.shape)
                det = detections[i]
                tid = track_ids[j]

                self.tracks[tid]["pos"] = det["bev"]
                self.tracks[tid]["missing"] = 0
                det["id"] = tid

                matched_dets.add(i)
                matched_tracks.add(j)
                D[i, :] = np.inf
                D[:, j] = np.inf

            for i, det in enumerate(detections):
                if i not in matched_dets:
                    self.tracks[self.next_id] = {"pos": det["bev"], "missing": 0}
                    det["id"] = self.next_id
                    self.next_id += 1

            for j, tid in enumerate(track_ids):
                if j not in matched_tracks:
                    self.tracks[tid]["missing"] += 1

        # Cleanup
        remove_ids = [tid for tid in self.tracks if self.tracks[tid]["missing"] > self.max_missing]
        for tid in remove_ids:
            del self.tracks[tid]
