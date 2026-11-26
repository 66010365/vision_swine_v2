"""Standalone ByteTrack-style tracker for TensorRT pig detections.

Uses NumPy and SciPy only; does not require PyTorch or Ultralytics.
Compatible with TensorRTByteTracker(settings).update(detections) in main.py.
"""

from __future__ import annotations


# ======================================================
# Import Libraries
# ======================================================

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# ======================================================
# Kalman Filter Configuration
# ======================================================

_STATE_SIZE = 8
_MEASURE_SIZE = 4
_POSITION_WEIGHT = 1.0 / 20.0
_VELOCITY_WEIGHT = 1.0 / 160.0


# ======================================================
# Convert Bounding Boxes
# ======================================================

def _xyxy_to_xyah(box):
    """Convert [x1, y1, x2, y2] to [cx, cy, aspect, height]."""

    x1, y1, x2, y2 = map(float, box)
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)

    return np.array(
        [(x1 + x2) / 2, (y1 + y2) / 2, width / height, height],
        dtype=np.float64,
    )


def _xyah_to_xyxy(state):
    """Convert [cx, cy, aspect, height] back to pixel coordinates."""

    cx, cy, aspect, height = state[:4]
    height = max(float(height), 1.0)
    width = max(float(aspect) * height, 1.0)

    return np.array(
        [cx - width / 2, cy - height / 2,
         cx + width / 2, cy + height / 2],
        dtype=np.float64,
    )


# ======================================================
# Kalman Filter
# ======================================================

class _KalmanFilter:
    """Constant-velocity Kalman filter in (x, y, aspect, height) space."""

    def __init__(self):

        self.motion = np.eye(_STATE_SIZE, dtype=np.float64)
        self.motion[:_MEASURE_SIZE, _MEASURE_SIZE:] = np.eye(
            _MEASURE_SIZE, dtype=np.float64,
        )

        self.observe = np.eye(
            _STATE_SIZE, dtype=np.float64,
        )[:_MEASURE_SIZE]

    def initiate(self, measurement):

        mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        height = max(float(measurement[3]), 1.0)

        std = np.array([
            2 * _POSITION_WEIGHT * height,
            2 * _POSITION_WEIGHT * height,
            1e-2,
            2 * _POSITION_WEIGHT * height,
            10 * _VELOCITY_WEIGHT * height,
            10 * _VELOCITY_WEIGHT * height,
            1e-5,
            10 * _VELOCITY_WEIGHT * height,
        ])

        return mean, np.diag(std ** 2)

    def predict(self, mean, covariance):

        height = max(float(mean[3]), 1.0)
        std = np.array([
            _POSITION_WEIGHT * height,
            _POSITION_WEIGHT * height,
            1e-2,
            _POSITION_WEIGHT * height,
            _VELOCITY_WEIGHT * height,
            _VELOCITY_WEIGHT * height,
            1e-5,
            _VELOCITY_WEIGHT * height,
        ])

        mean = self.motion @ mean
        covariance = (
            self.motion @ covariance @ self.motion.T
            + np.diag(std ** 2)
        )

        return mean, covariance

    def update(self, mean, covariance, measurement):

        height = max(float(mean[3]), 1.0)
        std = np.array([
            _POSITION_WEIGHT * height,
            _POSITION_WEIGHT * height,
            1e-1,
            _POSITION_WEIGHT * height,
        ])

        measurement_covariance = np.diag(std ** 2)
        projected_mean = self.observe @ mean
        projected_covariance = (
            self.observe @ covariance @ self.observe.T
            + measurement_covariance
        )

        # Solving a linear system avoids explicitly inverting the matrix.
        gain = np.linalg.solve(
            projected_covariance,
            self.observe @ covariance,
        ).T

        innovation = measurement - projected_mean
        mean = mean + gain @ innovation
        covariance = covariance - gain @ projected_covariance @ gain.T
        covariance = (covariance + covariance.T) / 2.0

        return mean, covariance


# ======================================================
# Track State
# ======================================================

@dataclass
class _Track:
    """One tracked pig, including its motion and latest detection index."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    last_seen: int
    state: str = "tracked"
    detection_index: int = -1

    def predict(self, kalman):

        # Ignore accumulated motion when a lost track is predicted.
        if self.state == "lost":
            self.mean[7] = 0.0

        self.mean, self.covariance = kalman.predict(
            self.mean, self.covariance,
        )

    def correct(self, kalman, detection, frame_number):

        self.mean, self.covariance = kalman.update(
            self.mean,
            self.covariance,
            _xyxy_to_xyah(detection["box"]),
        )

        self.score = detection["confidence"]
        self.last_seen = frame_number
        self.detection_index = detection["index"]
        self.state = "tracked"

    @property
    def box(self):
        return _xyah_to_xyxy(self.mean)


# ======================================================
# Bounding Box Similarity
# ======================================================

def _iou_matrix(tracks, detections):
    """Calculate intersection-over-union for all track/detection pairs."""

    if not tracks or not detections:
        return np.zeros(
            (len(tracks), len(detections)),
            dtype=np.float64,
        )

    track_boxes = np.stack([track.box for track in tracks])
    detection_boxes = np.stack([det["box"] for det in detections])

    top_left = np.maximum(
        track_boxes[:, None, :2],
        detection_boxes[None, :, :2],
    )
    bottom_right = np.minimum(
        track_boxes[:, None, 2:],
        detection_boxes[None, :, 2:],
    )

    overlap_size = np.maximum(bottom_right - top_left, 0)
    intersection = overlap_size[:, :, 0] * overlap_size[:, :, 1]

    track_size = np.maximum(track_boxes[:, 2:] - track_boxes[:, :2], 0)
    detection_size = np.maximum(
        detection_boxes[:, 2:] - detection_boxes[:, :2], 0,
    )

    track_area = track_size[:, 0] * track_size[:, 1]
    detection_area = detection_size[:, 0] * detection_size[:, 1]
    union = track_area[:, None] + detection_area[None, :] - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


# ======================================================
# Match Tracks to Detections
# ======================================================

def _match(tracks, detections, distance_limit, fuse_score=False):
    """Hungarian assignment with the ByteTrack IoU-distance convention."""

    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    similarities = _iou_matrix(tracks, detections)

    if fuse_score:
        scores = np.asarray(
            [detection["confidence"] for detection in detections],
            dtype=np.float64,
        )
        similarities = similarities * scores[None, :]

    distances = 1.0 - similarities
    # Invalid pairs have a large cost so Hungarian prefers valid pairs.
    costs = np.where(distances <= distance_limit, distances, 1e6)
    row_indices, column_indices = linear_sum_assignment(costs)

    matches = []
    used_tracks = set()
    used_detections = set()

    for row, column in zip(row_indices, column_indices):
        if costs[row, column] >= 1e6:
            continue

        matches.append((int(row), int(column)))
        used_tracks.add(int(row))
        used_detections.add(int(column))

    unmatched_tracks = [
        index for index in range(len(tracks))
        if index not in used_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections))
        if index not in used_detections
    ]

    return matches, unmatched_tracks, unmatched_detections


# ======================================================
# TensorRT ByteTrack Adapter
# ======================================================

class TensorRTByteTracker:
    """Two-stage ByteTrack-style association without Torch/Ultralytics.

    Input: list of {'box': [x1, y1, x2, y2], 'confidence': float, ...}.
    Output: {original_detection_index: (track_id, tracked_xyxy)}.

    Only detections actually matched this frame are returned. Lost-track
    predictions are internal and must not be counted as line crossings.
    """

    # ==================================================
    # Initialize Tracker
    # ==================================================

    def __init__(self, settings: dict):

        self.high_threshold = float(
            settings.get("track_high_thresh", 0.35)
        )
        self.low_threshold = float(
            settings.get("track_low_thresh", 0.10)
        )
        self.new_threshold = float(
            settings.get("new_track_thresh", 0.45)
        )
        self.match_threshold = float(
            settings.get("match_thresh", 0.80)
        )
        self.track_buffer = int(
            settings.get("track_buffer", 30)
        )
        self.fuse_score = bool(
            settings.get("fuse_score", True)
        )

        if not (
            0 <= self.low_threshold < self.high_threshold <= 1
        ):
            raise ValueError(
                "track_low_thresh must be below track_high_thresh"
            )

        if not (0 <= self.new_threshold <= 1):
            raise ValueError("new_track_thresh must be between 0 and 1")

        if not (0 <= self.match_threshold <= 1):
            raise ValueError("match_thresh must be between 0 and 1")

        if self.track_buffer < 1:
            raise ValueError("track_buffer must be at least 1")

        self.kalman = _KalmanFilter()
        self.tracks = []
        self.frame_number = 0
        self.next_track_id = 1

    # ==================================================
    # Prepare TensorRT Detections
    # ==================================================

    @staticmethod
    def _prepare_detections(detections):

        prepared = []

        for index, detection in enumerate(detections):
            try:
                box = np.asarray(detection["box"], dtype=np.float64)
                score = float(detection.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

            if box.shape != (4,) or not np.isfinite(box).all():
                continue

            if not np.isfinite(score) or not (0 <= score <= 1):
                continue

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            prepared.append({
                "index": index,
                "box": box,
                "confidence": score,
            })

        return prepared

    # ==================================================
    # Update Track with Detection
    # ==================================================

    def _correct_tracks(self, tracks, detections, matches):

        for track_index, detection_index in matches:
            tracks[track_index].correct(
                self.kalman,
                detections[detection_index],
                self.frame_number,
            )

    # ==================================================
    # Initialize New Track
    # ==================================================

    def _start_track(self, detection):

        mean, covariance = self.kalman.initiate(
            _xyxy_to_xyah(detection["box"])
        )

        track = _Track(
            track_id=self.next_track_id,
            mean=mean,
            covariance=covariance,
            score=detection["confidence"],
            last_seen=self.frame_number,
            detection_index=detection["index"],
        )

        self.next_track_id += 1
        self.tracks.append(track)

    # ==================================================
    # Update ByteTrack
    # ==================================================

    def update(self, detections: list[dict]) -> dict:
        """Advance one frame and return only observed, tracked pigs."""

        self.frame_number += 1
        all_detections = self._prepare_detections(detections)

        high_detections = [
            detection for detection in all_detections
            if detection["confidence"] >= self.high_threshold
        ]
        low_detections = [
            detection for detection in all_detections
            if self.low_threshold <= detection["confidence"]
            < self.high_threshold
        ]

        # Predict existing positions even when the detector returns nothing.
        for track in self.tracks:
            track.predict(self.kalman)
            track.detection_index = -1

        # First stage: associate both tracked and temporarily lost IDs with
        # confident detections, allowing reactivation of the same track ID.
        candidates = [
            track for track in self.tracks
            if track.state in ("tracked", "lost")
        ]
        matches, unmatched_candidates, unmatched_high = _match(
            candidates,
            high_detections,
            self.match_threshold,
            fuse_score=self.fuse_score,
        )
        self._correct_tracks(candidates, high_detections, matches)

        # Second stage: salvage existing active tracks using weaker detections.
        # Lost tracks do not use low-confidence observations to reactivate.
        remaining_tracked = [
            candidates[index]
            for index in unmatched_candidates
            if candidates[index].state == "tracked"
        ]
        second_matches, _, _ = _match(
            remaining_tracked,
            low_detections,
            distance_limit=0.50,
            fuse_score=False,
        )
        self._correct_tracks(
            remaining_tracked, low_detections, second_matches
        )

        # Unmatched tracks remain recoverable for track_buffer frames.
        for track in self.tracks:
            if track.last_seen != self.frame_number:
                track.state = "lost"

        # Start new IDs only from high-confidence, unmatched detections.
        for detection_index in unmatched_high:
            detection = high_detections[detection_index]
            if detection["confidence"] >= self.new_threshold:
                self._start_track(detection)

        # Expire lost tracks after the configured retention interval.
        self.tracks = [
            track for track in self.tracks
            if self.frame_number - track.last_seen <= self.track_buffer
        ]

        # Return only detections observed in this frame to the counter.
        assignments = {}
        for track in self.tracks:
            if (
                track.state == "tracked"
                and track.last_seen == self.frame_number
                and track.detection_index >= 0
            ):
                assignments[track.detection_index] = (
                    track.track_id,
                    track.box.astype(float).tolist(),
                )

        return assignments
