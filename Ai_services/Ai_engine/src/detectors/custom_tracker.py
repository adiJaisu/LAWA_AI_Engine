import os
import cv2
import numpy as np
import torch
import torchreid
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from collections import deque
from pathlib import Path

# ============================================================================
# BYTETRACK TRACKER
# ============================================================================

class ByteTrackTracker:
    def __init__(self, track_thresh, track_buffer, match_thresh, min_tracklet_len, embedding_weight, iou_weight):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_tracklet_len = min_tracklet_len
        self.embedding_weight = embedding_weight
        self.iou_weight = iou_weight
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_id_count = 0

    def update(self, detections, scores, embeddings):
        self.frame_id += 1

        scores = np.array(scores)
        remain_inds = scores > self.track_thresh
        dets = [detections[i] for i in range(len(detections)) if remain_inds[i]]
        scores_keep = [scores[i] for i in range(len(scores)) if remain_inds[i]]
        embeddings_keep = [embeddings[i] for i in range(len(embeddings)) if remain_inds[i]]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track['is_activated']:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        matches, u_track, u_detection = self._associate(
            tracked_stracks, dets, embeddings_keep
        )

        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det = dets[idet]
            track['bbox'] = det
            track['embedding'] = embeddings_keep[idet]
            track['score'] = scores_keep[idet]
            track['tracklet_len'] += 1
            track['frame_id'] = self.frame_id
            track['is_activated'] = True

        for inew in u_detection:
            track = self._init_track(dets[inew], scores_keep[inew], embeddings_keep[inew])
            self.tracked_stracks.append(track)

        for it in u_track:
            track = tracked_stracks[it]
            track['state'] = 'lost'
            self.lost_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t['state'] == 'tracked']
        self.lost_stracks = [
            t for t in self.lost_stracks 
            if self.frame_id - t['frame_id'] <= self.track_buffer
        ]

        output = []
        for track in self.tracked_stracks:
            if track['is_activated'] and track['tracklet_len'] >= self.min_tracklet_len:
                output.append((
                    track['track_id'],
                    track['bbox'],
                    track['embedding']
                ))
        
        return output

    def _associate(self, tracks, detections, embeddings):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, (det, emb) in enumerate(zip(detections, embeddings)):
                iou = self._iou(track['bbox'], det)
                emb_dist = self._cosine_distance(track['embedding'], emb)
                cost_matrix[i, j] = self.embedding_weight * emb_dist + self.iou_weight * (1 - iou)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def _init_track(self, bbox, score, embedding):
        self.track_id_count += 1
        return {
            'track_id': self.track_id_count,
            'bbox': bbox,
            'embedding': embedding,
            'score': score,
            'tracklet_len': 1,
            'state': 'tracked',
            'is_activated': True,
            'frame_id': self.frame_id
        }

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)

    def _cosine_distance(self, emb1, emb2):
        return cosine(emb1, emb2)

# ============================================================================
# REID MODEL
# ============================================================================

class ReIDModel:
    def __init__(self, model_name, num_classes, image_size, normalize_mean, normalize_std, device='auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available. Set device='cpu'.")

        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=True
        )

        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(normalize_mean, normalize_std)
        ])

    def extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img)

        feat = feat.cpu().numpy()[0]
        feat = feat / np.linalg.norm(feat)
        return feat

# ============================================================================
# IDENTITY MANAGER
# ============================================================================

class IdentityManager:
    def __init__(self, reid_model, params):
        self.reid_model = reid_model

        self.threshold = params.get("threshold", 0.5)
        self.avg_threshold = params.get("avg_threshold", 0.5)
        self.keep_threshold = params.get("keep_threshold", 0.35)
        self.color_threshold = params.get("color_threshold", 0.35)
        self.reactivation_window = params.get("reactivation_window", 200)
        self.reactivation_centroid_thresh = params.get("reactivation_centroid_thresh", 0.55)
        self.reactivation_color_thresh = params.get("reactivation_color_thresh", 0.45)
        self.long_absence_window = params.get("long_absence_window", 1000)
        self.long_centroid_thresh = params.get("long_centroid_thresh", 0.7)
        self.long_color_thresh = params.get("long_color_thresh", 0.55)
        self.save_dir = params.get("save_dir", "identities")
        self.max_embeddings = params.get("max_embeddings", 40)
        self.max_color_hists = params.get("max_color_hists", 15)

        self.score_centroid_weight = params.get("score_centroid_weight", 0.5)
        self.score_avg_weight = params.get("score_avg_weight", 0.3)
        self.score_color_weight = params.get("score_color_weight", 0.2)
        
        self.color_hist_bins = [8, 8, 8]
        self.color_hist_ranges = [0, 180, 0, 256, 0, 256]
        
        self.identities = {}
        self.next_identity_id = 1
        os.makedirs(self.save_dir, exist_ok=True)

    def clear_saved_identities(self):
        for name in os.listdir(self.save_dir):
            path = os.path.join(self.save_dir, name)
            if os.path.isdir(path):
                for root, _, files in os.walk(path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    os.rmdir(root)
        self.identities.clear()
        self.next_identity_id = 1

    def assign_identity(self, track_id, crop, frame_index, preferred_identity=None, exclude_ids=None):
        if crop is None or crop.size == 0:
            return None

        current_embedding = self._normalize(self.reid_model.extract(crop))

        if len(self.identities) == 0:
            return self._create_identity(current_embedding, crop, track_id, frame_index)

        if exclude_ids is None:
            exclude_ids = set()

        metrics = {}
        for identity_id, data in self.identities.items():
            if identity_id in exclude_ids:
                continue
            if data["active"] and data["current_track"] != track_id:
                continue

            centroid_dist, avg_dist, color_dist = self._compute_distances(current_embedding, crop, data)

            time_since_seen = frame_index - data["last_seen"] if data["last_seen"] >= 0 else float("inf")
            recent = time_since_seen <= self.reactivation_window
            ignore_color = not recent

            centroid_limit = self.threshold
            color_limit = self.color_threshold
            if recent:
                centroid_limit = self.reactivation_centroid_thresh
                color_limit = self.reactivation_color_thresh
            else:
                centroid_limit = self.long_centroid_thresh
                color_limit = 1.0 if ignore_color else self.long_color_thresh

            metrics[identity_id] = {
                "centroid": centroid_dist,
                "avg": avg_dist,
                "color": color_dist,
                "score": (self.score_centroid_weight * centroid_dist + 
                         self.score_avg_weight * avg_dist + 
                         self.score_color_weight * color_dist),
                "centroid_limit": centroid_limit,
                "color_limit": color_limit,
                "ignore_color": ignore_color,
            }

        if preferred_identity is not None and preferred_identity in metrics:
            pref = metrics[preferred_identity]
            if pref["centroid"] <= self.keep_threshold and pref["color"] <= self.color_threshold:
                self._update_identity(preferred_identity, current_embedding, crop)
                self._mark_active(preferred_identity, track_id, frame_index)
                return preferred_identity
            elif (
                pref["centroid"] <= pref["centroid_limit"]
                and pref["avg"] <= self.avg_threshold
                and pref["color"] <= pref["color_limit"]
            ):
                self._update_identity(preferred_identity, current_embedding, crop)
                self._mark_active(preferred_identity, track_id, frame_index)
                return preferred_identity

        best_identity = None
        best_score = float("inf")
        for identity_id, info in metrics.items():
            if (
                info["centroid"] <= info["centroid_limit"]
                and info["avg"] <= self.avg_threshold
                and info["color"] <= info["color_limit"]
            ):
                if info["score"] < best_score:
                    best_score = info["score"]
                    best_identity = identity_id

        if best_identity is not None:
            self._update_identity(best_identity, current_embedding, crop)
            self._mark_active(best_identity, track_id, frame_index)
            return best_identity

        new_id = self._create_identity(current_embedding, crop, track_id, frame_index)
        return new_id
    
    def _create_identity(self, embedding, crop, track_id, frame_index):
        identity_id = self.next_identity_id
        self.next_identity_id += 1

        self.identities[identity_id] = {
            "embeddings": deque([embedding], maxlen=self.max_embeddings),
            "centroid": embedding.copy(),
            "color_hists": deque([self._extract_color_hist(crop)], maxlen=self.max_color_hists),
            "color_centroid": self._extract_color_hist(crop),
            "crop_count": 0,
            "active": True,
            "current_track": track_id,
            "last_seen": frame_index,
        }

        self._save_crop(identity_id, crop)
        return identity_id

    def _update_identity(self, identity_id, embedding, crop):
        entry = self.identities[identity_id]
        entry["embeddings"].append(embedding)
        entry["centroid"] = self._normalize(np.mean(entry["embeddings"], axis=0))
        if crop is not None and crop.size != 0:
            color_hist = self._extract_color_hist(crop)
            entry["color_hists"].append(color_hist)
            entry["color_centroid"] = self._normalize(np.mean(entry["color_hists"], axis=0))
        self._save_crop(identity_id, crop)

    def _mark_active(self, identity_id, track_id, frame_index):
        entry = self.identities.get(identity_id)
        if entry is None:
            return
        entry["active"] = True
        entry["current_track"] = track_id
        entry["last_seen"] = frame_index

    def mark_inactive(self, identity_id):
        entry = self.identities.get(identity_id)
        if entry is None:
            return
        entry["active"] = False
        entry["current_track"] = None

    def _compute_distances(self, embedding, crop, entry):
        centroid_dist = cosine(embedding, entry["centroid"])
        if len(entry["embeddings"]) == 0:
            avg_dist = centroid_dist
        else:
            distances = [cosine(embedding, stored) for stored in entry["embeddings"]]
            avg_dist = float(np.mean(distances)) if distances else centroid_dist

        if crop is None or crop.size == 0:
            color_dist = 1.0
        else:
            color_hist = self._extract_color_hist(crop)
            color_dist = cosine(color_hist, entry["color_centroid"])

        return centroid_dist, avg_dist, color_dist

    def _save_crop(self, identity_id, crop):
        if crop is None or crop.size == 0:
            return

        identity_dir = os.path.join(self.save_dir, f"person_{identity_id}")
        os.makedirs(identity_dir, exist_ok=True)

        entry = self.identities[identity_id]
        crop_index = entry["crop_count"]
        entry["crop_count"] += 1

        path = os.path.join(identity_dir, f"frame_{crop_index:05d}.jpg")
        cv2.imwrite(path, crop)

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector) + 1e-8
        return vector / norm

    def _extract_color_hist(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.color_hist_bins, self.color_hist_ranges)
        hist = cv2.normalize(hist, hist).flatten()
        norm = np.linalg.norm(hist) + 1e-8
        return hist / norm

# ============================================================================
# PERSON TRACKER
# ============================================================================

class CustomPersonTracker:
    def __init__(self, yolo_detector, tracking_params, identity_params, reid_params, person_class_id=0):
        self.detector = yolo_detector.model  # Use the YOLO model directly
        
        self.reid = ReIDModel(
            model_name=reid_params.get("model_name", "osnet_x1_0"),
            num_classes=reid_params.get("num_classes", 1000),
            image_size=reid_params.get("image_size", (256, 128)),
            normalize_mean=reid_params.get("normalize_mean", [0.485, 0.456, 0.406]),
            normalize_std=reid_params.get("normalize_std", [0.229, 0.224, 0.225]),
            device=reid_params.get("device", "auto")
        )
        self.tracker = ByteTrackTracker(
            track_thresh=tracking_params.get("track_thresh", 0.5),
            track_buffer=tracking_params.get("track_buffer", 100),
            match_thresh=tracking_params.get("match_thresh", 0.8),
            min_tracklet_len=tracking_params.get("min_tracklet_len", 2),
            embedding_weight=tracking_params.get("embedding_weight", 0.7),
            iou_weight=tracking_params.get("iou_weight", 0.3)
        )
        self.identity_manager = IdentityManager(self.reid, identity_params)
        
        self.track_to_identity = {}
        self.frame_index = 0
        self.person_class_id = person_class_id

    def process_frame(self, frame, classes_id, confidence, img_size=640, device="cpu"):
        self.frame_index += 1
        results = self.detector(frame)[0]

        boxes = []
        scores = []
        embeddings = []

        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) != self.person_class_id:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            embedding = self.reid.extract(crop)

            boxes.append((x1, y1, x2, y2))
            scores.append(float(conf))
            embeddings.append(embedding)

        confirmed_tracks = self.tracker.update(boxes, scores, embeddings)

        tracked_boxes = []
        tracked_ids = []

        used_identity_ids = set()
        current_track_ids = set()

        for track_id, bbox, embedding in confirmed_tracks:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if track_id in self.track_to_identity:
                identity_id = self.identity_manager.assign_identity(
                    track_id,
                    crop,
                    self.frame_index,
                    preferred_identity=self.track_to_identity[track_id],
                    exclude_ids=used_identity_ids,
                )
                self.track_to_identity[track_id] = identity_id
            else:
                identity_id = self.identity_manager.assign_identity(
                    track_id,
                    crop,
                    self.frame_index,
                    exclude_ids=used_identity_ids,
                )
                self.track_to_identity[track_id] = identity_id

            if identity_id in used_identity_ids:
                identity_id = self.identity_manager.assign_identity(
                    track_id,
                    crop,
                    self.frame_index,
                    exclude_ids=used_identity_ids,
                )
                self.track_to_identity[track_id] = identity_id

            used_identity_ids.add(identity_id)
            current_track_ids.add(track_id)

            tracked_boxes.append((x1, y1, x2, y2))
            tracked_ids.append(identity_id)

        stale_tracks = [tid for tid in self.track_to_identity.keys() if tid not in current_track_ids]
        for tid in stale_tracks:
            identity_id = self.track_to_identity.pop(tid, None)
            if identity_id is not None:
                self.identity_manager.mark_inactive(identity_id)

        return tracked_boxes, tracked_ids
