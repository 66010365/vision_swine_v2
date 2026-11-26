# ======================================================
# Import Libraries
# ======================================================

import cv2
import time
import threading
import queue
import logging
import json
import os
import shutil
import math
from pathlib import Path

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ======================================================
# Import Project Modules
# ======================================================

from vision_swine.runtime_config import load_runtime_config
from vision_swine.tracking import TensorRTByteTracker

from vision_swine.detector import PigDetector
from vision_swine.features import extract_features
from vision_swine.weight_predictor import predict_weight
from vision_swine.api_client import create_session, end_session, send_weight, send_counting_event, API_BASE_URL
from vision_swine.video_recorder import SegmentedVideoRecorder
from vision_swine.video_uploader import VideoUploadWorker


# ======================================================
# Configuration
# ======================================================

RUNTIME_CONFIG = load_runtime_config()
CAMERA_INDEX = RUNTIME_CONFIG['camera'].get('source', 0)
CAMERA_WIDTH = int(RUNTIME_CONFIG['camera'].get('width', 640))
CAMERA_HEIGHT = int(RUNTIME_CONFIG['camera'].get('height', 480))
CAMERA_FPS = int(RUNTIME_CONFIG['camera'].get('target_fps', 30))
PROCESS_EVERY = int(RUNTIME_CONFIG['inference']['process_every'])  # Enforced as 1 by runtime_config

# Estimate weights less frequently than detection/tracking. Optional YAML overrides:
# weight_estimation:
#   interval_frames: 10
#   max_updates_per_frame: 2
WEIGHT_SETTINGS = RUNTIME_CONFIG.get('weight_estimation', {})
WEIGHT_INTERVAL_FRAMES = max(1, int(WEIGHT_SETTINGS.get('interval_frames', 10)))
MAX_WEIGHT_UPDATES_PER_FRAME = max(1, int(WEIGHT_SETTINGS.get('max_updates_per_frame', 2)))
# Limit database writes; this is separate from the model's weight-estimation interval.
WEIGHT_UPLOAD_INTERVAL_FRAMES = 300
API_DEVICE_ID = 1  # Existing devices.device_id in Vision Swine V2
STREAM_HOST = RUNTIME_CONFIG['stream'].get('host', '0.0.0.0')
STREAM_PORT = int(RUNTIME_CONFIG['stream'].get('port', 8080))
API_VERSION = 'gate-config-v3-bytetrack'
JPEG_QUALITY = int(RUNTIME_CONFIG['stream'].get('jpeg_quality', 80))

# ======================================================
# Local MP4 Recording Settings
# ======================================================

RECORDING_DIRECTORY = RUNTIME_CONFIG['_project_root'] / "outputs" / "recordings"
RECORDING_FPS = 15.0
RECORDING_SEGMENT_SECONDS = 180

WINDOW_NAME = "Vision Swine"


# ======================================================
# Logging Configuration
# ======================================================

logging.basicConfig(
    level=getattr(logging, str(RUNTIME_CONFIG.get('logging', {}).get('level', 'INFO')).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ======================================================
# Initialize Shared Stream State
# ======================================================

latest_frame = None
frame_lock = threading.Lock()
# Raw camera frames are captured independently of AI inference so the live stream
# does not stall while the detector is working.
camera_frame_lock = threading.Lock()
latest_camera_frame = None
latest_camera_seq = 0
detection_lock = threading.Lock()
latest_detection_results = []
latest_track_info = {}
latest_predicted_boxes = []
state_lock = threading.Lock()
latest_pig_count = 0
latest_in_count = 0
latest_out_count = 0
latest_net_count = 0
latest_fps = 0.0
latest_processing_ms = 0.0
network_previous = None
network_speeds = {"rx_speed": 0.0, "tx_speed": 0.0}
network_last_update = time.time()
running = True

# Shared gate configuration. Persisted separately from the optional legacy app.yaml.
PROJECT_ROOT = RUNTIME_CONFIG['_project_root']
GATE_CONFIG_FILE = RUNTIME_CONFIG['counting']['resolved_gate_config']
GATE_CONFIG_LOCK = threading.Lock()
active_counter = None


# ======================================================
# Open Camera
# ======================================================

def open_camera(stop_on_shutdown=False, camera=None):
    """
    Detect the USB camera by its stable device ID.
    Retry until the camera is available and frames can be read.
    """

    camera_path = (
        "/dev/v4l/by-id/"
        "usb-4K_USB_Camera_4K_USB_Camera_01.00.00-video-index0"
    )

    attempt = 0

    while True:
        if stop_on_shutdown and not running:
            return None

        attempt += 1

        if camera is None:
            camera = cv2.VideoCapture(
                camera_path,
                cv2.CAP_V4L2,
            )
        else:
            camera.open(camera_path, cv2.CAP_V4L2)

        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = camera.read()

            if ret and frame is not None:
                logger.info(
                    "Camera opened successfully: %s",
                    camera_path,
                )
                return camera

        camera.release()

        logger.warning(
            "Camera not ready (attempt %d). Retrying in 5 seconds...",
            attempt,
        )

        if stop_on_shutdown:
            for _ in range(50):
                if not running:
                    return None
                time.sleep(0.1)
        else:
            time.sleep(5)


# ======================================================
# Load AI Model
# ======================================================

def load_model():
    """
    Initialize the TensorRT pig detection model.
    """

    logger.info("Loading AI model...")

    detector = PigDetector()

    logger.info('AI model loaded successfully: %s', RUNTIME_CONFIG['model']['resolved_path'])

    return detector


# ======================================================
# Estimate Pig Weight
# ======================================================

def estimate_weight(mask):
    """
    Extract features from a segmentation mask
    and predict the pig's weight.
    """

    if mask is None:
        return None

    try:
        features = extract_features(mask)

        if features is None:
            return None

        weight = predict_weight(features)

        return float(weight)

    except Exception:
        logger.exception("Weight estimation failed")

        return None


# ======================================================
# Line-crossing counter using ByteTrack IDs
# ======================================================

def load_counting_config():
    """Load gate/tracking/counting settings from YAML; gate.json stores UI edits only."""
    cfg = RUNTIME_CONFIG
    counting = cfg['counting']
    tracking = cfg['tracking']
    gate = counting.get('gate', {})
    lane = counting.get('lane', {})
    defaults = {
        'line_y': int(gate.get('line_y', 178)),
        'x_left': int(gate.get('x_left', 130)),
        'x_right': int(gate.get('x_right', 490)),
        'margin': int(gate.get('margin', 20)),
        'lane_x1': int(lane.get('x1', 93)),
        'lane_y1': int(lane.get('y1', 43)),
        'lane_x2': int(lane.get('x2', 541)),
        'lane_y2': int(lane.get('y2', 333)),
        'stable_frames': int(tracking.get('stable_frames', 3)),
        'max_miss_frames': int(tracking.get('max_miss_frames', 30)),
        'cooldown_seconds': float(counting.get('cooldown_seconds', 0.30)),
        'in_direction': str(counting.get('direction', {}).get('in', 'BELOW_TO_ABOVE')).upper(),
        'out_direction': str(counting.get('direction', {}).get('out', 'ABOVE_TO_BELOW')).upper(),
    }
    try:
        with GATE_CONFIG_LOCK:
            if GATE_CONFIG_FILE.exists():
                with GATE_CONFIG_FILE.open('r', encoding='utf-8') as fh:
                    saved = json.load(fh)
                defaults['x_left'] = int(saved.get('x_left', defaults['x_left']))
                defaults['x_right'] = int(saved.get('x_right', defaults['x_right']))
                defaults['line_y'] = int(saved.get('line_y', saved.get('main_gate_y', defaults['line_y'])))
                defaults['margin'] = int(saved.get('margin', saved.get('margin_offset', defaults['margin'])))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning('Could not load gate config %s: %s', GATE_CONFIG_FILE, exc)
    return defaults


def save_gate_config(payload):
    """Validate and persist gate settings, then apply them to the live counter."""
    global active_counter

    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")

    current = load_counting_config()
    aliases = {
        "x_left": ("x_left",),
        "x_right": ("x_right",),
        "line_y": ("line_y", "main_gate_y"),
        "margin": ("margin", "margin_offset"),
    }
    updated = {}
    for target, keys in aliases.items():
        value = next((payload[key] for key in keys if key in payload), current[target])
        if isinstance(value, bool):
            raise ValueError(f"{target} must be an integer")
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{target} must be a finite integer")
        updated[target] = int(number)

    if updated["x_left"] < 0 or updated["x_right"] <= updated["x_left"]:
        raise ValueError("x_left must be >= 0 and smaller than x_right")
    if updated["line_y"] < 0 or updated["margin"] < 0:
        raise ValueError("line_y and margin must be >= 0")

    GATE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GATE_CONFIG_LOCK:
        with GATE_CONFIG_FILE.open("w", encoding="utf-8") as fh:
            json.dump(updated, fh, ensure_ascii=False, indent=2)

        counter = active_counter
        if counter is not None:
            with counter.lock:
                counter.cfg.update(updated)

    logger.info("Gate configuration saved: %s", updated)
    return updated


class LineCrossingCounter:
    """Count only confirmed ByteTrack IDs crossing stable sides of the gate.

    Tracking itself occurs over the whole image, so an ID survives entering or
    leaving the counting lane. Predicted boxes are visual hints only and NEVER
    contribute to counts.
    """

    def __init__(self):
        self.cfg = load_counting_config()
        self.lock = threading.RLock()
        self.tracker = TensorRTByteTracker(RUNTIME_CONFIG['tracking'])
        self.tracks = {}
        self.in_count = 0
        self.out_count = 0
        self.net_count = 0
        self.frame_idx = 0
        self.predicted_boxes = []
        self.prediction_ttl = max(0, int(RUNTIME_CONFIG['tracking'].get('prediction_frames', 2)))

    @staticmethod
    def center(box):
        x1, y1, x2, y2 = map(float, box)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _update_tracking_and_counting(self, detections):
        with self.lock:
            self.frame_idx += 1
            cfg = self.cfg
            # ByteTrack is the only source of IDs. No centroid association.
            assignments = self.tracker.update(detections)
            events = []
            info = {}
            now = time.monotonic()

            for idx, (tid, tracked_box) in assignments.items():
                cx, cy = self.center(tracked_box)
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = {
                        'center': (cx, cy), 'box': list(tracked_box),
                        'velocity': (0.0, 0.0), 'last_seen': self.frame_idx,
                        'side': None, 'candidate_side': None, 'stable': 0,
                        'last_crossing': -float('inf'), 'label': None,
                        'counted': False,
                    }
                    self.tracks[tid] = tr
                else:
                    dt = max(1, self.frame_idx - tr['last_seen'])
                    tx, ty = tr['center']
                    tr['velocity'] = ((cx - tx) / dt, (cy - ty) / dt)
                    tr['center'] = (cx, cy)
                    tr['box'] = list(tracked_box)
                    tr['last_seen'] = self.frame_idx

                info[idx] = (tid, tr)
                eligible = (
                    cfg['lane_x1'] <= cx <= cfg['lane_x2']
                    and cfg['lane_y1'] <= cy <= cfg['lane_y2']
                    and cfg['x_left'] <= cx <= cfg['x_right']
                )
                if not eligible:
                    tr['candidate_side'] = None
                    tr['stable'] = 0
                    continue

                if cy < cfg['line_y'] - cfg['margin']:
                    side = 'ABOVE'
                elif cy > cfg['line_y'] + cfg['margin']:
                    side = 'BELOW'
                else:
                    side = 'CROSS'
                if side == 'CROSS':
                    # Retain the last confirmed side while within the gate margin.
                    tr['candidate_side'] = None
                    tr['stable'] = 0
                    continue
                if tr['candidate_side'] == side:
                    tr['stable'] += 1
                else:
                    tr['candidate_side'] = side
                    tr['stable'] = 1
                if tr['stable'] < max(1, cfg['stable_frames']):
                    continue

                previous_side = tr['side']
                if previous_side is not None and previous_side != side:
                    transition = f'{previous_side}_TO_{side}'
                    direction = None
                    if transition == cfg['in_direction']:
                        direction = 'IN'
                    elif transition == cfg['out_direction']:
                        direction = 'OUT'
                    if direction and now - tr['last_crossing'] >= cfg['cooldown_seconds']:
                        if direction == 'IN':
                            self.in_count += 1
                            self.net_count += 1
                        else:
                            self.out_count += 1
                            self.net_count -= 1
                        tr['last_crossing'] = now
                        tr['label'] = direction
                        tr['counted'] = True
                        events.append({
                            'track_id': tid, 'direction': direction,
                            'in_count': self.in_count, 'out_count': self.out_count,
                            'net_count': self.net_count, 'timestamp': time.time(),
                        })
                tr['side'] = side

            # If the detector misses a pig, briefly show an explicitly labelled
            # predicted box. This is NOT a detection and NOT counted.
            self.predicted_boxes = []
            assigned_ids = {tid for tid, _ in assignments.values()}
            expired = []
            for tid, tr in self.tracks.items():
                missing = self.frame_idx - tr['last_seen']
                if missing > cfg['max_miss_frames']:
                    expired.append(tid)
                elif tid not in assigned_ids and 0 < missing <= self.prediction_ttl:
                    vx, vy = tr['velocity']
                    x1, y1, x2, y2 = tr['box']
                    self.predicted_boxes.append((
                        tid, [x1 + missing*vx, y1 + missing*vy,
                              x2 + missing*vx, y2 + missing*vy],
                    ))
            for tid in expired:
                del self.tracks[tid]
            return info, events

    def _attach_cached_weights(self, detections, track_info):
        """Estimate weight sparingly, retaining the last value for each Track ID.

        Tracking and line crossing run before feature extraction, so heavy
        skeleton calculations do not delay the current frame's count decision.
        Only detections matched by ByteTrack receive estimated weights.
        """
        frame_number = self.frame_idx
        updates = 0

        # By default, initialize/refresh at most two pigs on a single frame.
        # This avoids a very long pause when many tracks appear together.
        for detection_index, (track_id, track) in track_info.items():
            if detection_index >= len(detections):
                continue

            detection = detections[detection_index]
            detection['track_id'] = track_id
            last_attempt = track.get('weight_last_frame', -WEIGHT_INTERVAL_FRAMES)
            cached_weight = track.get('weight')

            due = frame_number - last_attempt >= WEIGHT_INTERVAL_FRAMES
            if due and updates < MAX_WEIGHT_UPDATES_PER_FRAME:
                mask = detection.get('mask')
                if mask is not None:
                    # Mark even failed attempts to avoid repeating costly
                    # feature extraction every single frame.
                    track['weight_last_frame'] = frame_number
                    updates += 1
                    weight = estimate_weight(mask)
                    if weight is not None and math.isfinite(weight):
                        track['weight'] = weight
                        cached_weight = weight

            detection['weight'] = cached_weight

    def update(self, detections):
        """Track/count every frame, then refresh cached weights as needed."""
        track_info, events = self._update_tracking_and_counting(detections)
        self._attach_cached_weights(detections, track_info)
        return track_info, events


# ======================================================
# Process Camera Frame
# ======================================================

def process_frame(frame, detector):
    """
    Detect and segment pigs on the current frame.

    The LineCrossingCounter.update() method associates detections with
    Track IDs and fills in cached weights *after* line-crossing decisions.
    This keeps weight feature extraction off the every-frame hot path.
    """
    results = []

    for detection in detector.detect(frame):
        result = dict(detection)
        result['weight'] = None
        results.append(result)

    return results


# ======================================================
# Draw Detection Results
# ======================================================

def draw_results(frame, results, track_info=None, counter=None, predicted_boxes=None):
    """Draw count gate, pig boxes and legible weight labels.

    Labels have an opaque dark background, short text, and are placed to
    minimize overlap.  This only changes rendering, not detection, tracking,
    counting or weight estimation.
    """
    output = frame.copy()
    height, width = output.shape[:2]
    track_info = track_info or {}
    labels = []

    for index, result in enumerate(results):
        box = result.get('box')
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        tid, _track = track_info.get(index, (None, {}))
        weight = result.get('weight')
        if weight is not None and math.isfinite(float(weight)):
            label = '#{} {:.1f}kg'.format(tid if tid is not None else index + 1,
                                         float(weight))
            color = (80, 230, 95)  # BGR: confirmed estimated weight
        else:
            label = '#{} --kg'.format(tid if tid is not None else index + 1)
            color = (175, 185, 195)  # BGR: not yet estimated
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        labels.append((label, color, (x1, y1, x2, y2)))

    # Reserve the top of the video for the external IN/OUT overlay.
    used_rectangles = [(0, 0, width, 34)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    pad_x, pad_y = 4, 3

    # Position text close to the pig; choose among nearby spots to avoid
    # other text rectangles. At high pig density overlap may still be
    # unavoidable within a 640x480 frame.
    for label, color, (x1, y1, x2, y2) in labels:
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        label_w = tw + 2 * pad_x
        label_h = th + baseline + 2 * pad_y
        cx = (x1 + x2) // 2
        positions = [
            (x1, y1 - label_h - 2),
            (cx - label_w // 2, y1 - label_h - 2),
            (x1, y1 + 2),
            (cx - label_w // 2, y2 + 2),
            (x1, y2 + 2),
            (x2 + 3, y1),
            (x1 - label_w - 3, y1),
            (cx - label_w // 2, (y1 + y2 - label_h) // 2),
        ]
        # Add horizontal lanes as fallback when detections crowd together.
        for y in range(36, max(37, height - label_h), label_h + 3):
            positions.append((cx - label_w // 2, y))

        best = None
        best_cost = float('inf')
        for px, py in positions:
            px = max(0, min(px, max(0, width - label_w)))
            py = max(34, min(py, max(34, height - label_h)))
            rect = (px, py, px + label_w, py + label_h)
            overlaps = 0
            for ox1, oy1, ox2, oy2 in used_rectangles:
                overlaps += max(0, min(rect[2], ox2) - max(px, ox1)) * \
                            max(0, min(rect[3], oy2) - max(py, oy1))
            distance = abs(px + label_w // 2 - cx) + abs(py - y1)
            cost = overlaps * 100 + distance * 0.1
            if cost < best_cost:
                best_cost = cost
                best = rect
            if overlaps == 0 and distance < 25:
                break

        left, top, right, bottom = best
        used_rectangles.append(best)
        cv2.rectangle(output, (left, top), (right, bottom), (18, 24, 28), -1)
        cv2.rectangle(output, (left, top), (right, bottom), color, 1)
        cv2.putText(output, label,
                    (left + pad_x, top + pad_y + th),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Show briefly extrapolated tracks in a muted color; never give them a
    # fake weight label.
    for tid, box in (predicted_boxes or []):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 1)

    if counter is not None:
        with counter.lock:
            gate = dict(counter.cfg)
        cv2.line(output, (gate['x_left'], gate['line_y']),
                 (gate['x_right'], gate['line_y']), (0, 165, 255), 2, cv2.LINE_AA)

    return output


# ======================================================
# Encode Frame for MJPEG Stream
# ======================================================

def encode_frame(frame):
    """
    Convert the annotated frame into JPEG bytes.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        return None

    return encoded.tobytes()


# ======================================================
# Publish MJPEG Stream
# ======================================================

def publish_frame(frame):
    """
    Update the shared JPEG frame for HTTP clients.
    """

    global latest_frame

    jpeg_data = encode_frame(frame)

    if jpeg_data is None:
        return

    with frame_lock:
        latest_frame = jpeg_data


# ======================================================
# Stream HTTP Handler
# ======================================================

class StreamHandler(BaseHTTPRequestHandler):
    """Serve MJPEG and the lightweight API used by the Dashboard."""

    def end_headers(self):
        origin = self.headers.get("Origin", "")
        if origin.startswith("http://100.") or origin in (
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ):
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            page = """<!DOCTYPE html><html><head><meta charset="utf-8">
            <title>Vision Swine</title></head><body>
            <h2>Vision Swine - Live AI Camera</h2>
            <img src="/stream" style="width:100%;max-width:960px;">
            </body></html>"""
            self.wfile.write(page.encode("utf-8"))
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while running:
                    with frame_lock:
                        jpeg_data = latest_frame
                    if jpeg_data is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg_data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        if path == "/api/health":
            self._send_json({
                "success": True,
                "service": "vision_swine_camera",
                "api_version": API_VERSION,
                "module_file": str(Path(__file__).resolve()),
                "gate_config_file": str(GATE_CONFIG_FILE),
            })
            return

        if path == "/api/system/realtime":
            self._send_json({"data": get_realtime_payload()})
            return

        if path == "/api/config/gate":
            cfg = load_counting_config()
            self._send_json({
                "x_left": cfg["x_left"],
                "x_right": cfg["x_right"],
                "line_y": cfg["line_y"],
                "main_gate_y": cfg["line_y"],
                "margin": cfg["margin"],
                "margin_offset": cfg["margin"],
                "enabled": True,
            })
            return

        if path == "/api/history":
            self._send_json({"data": []})
            return

        self._send_json({"error": "Not Found", "path": path}, status=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/config/gate":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    self._send_json({"error": "Request body is empty or too large"}, status=400)
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                cfg = save_gate_config(payload)
                self._send_json({
                    "success": True,
                    "message": "Gate settings saved",
                    "api_version": API_VERSION,
                    "x_left": cfg["x_left"],
                    "x_right": cfg["x_right"],
                    "line_y": cfg["line_y"],
                    "main_gate_y": cfg["line_y"],
                    "margin": cfg["margin"],
                    "margin_offset": cfg["margin"],
                    "data": {
                        "x_left": cfg["x_left"],
                        "x_right": cfg["x_right"],
                        "line_y": cfg["line_y"],
                        "main_gate_y": cfg["line_y"],
                        "margin": cfg["margin"],
                        "margin_offset": cfg["margin"],
                        "enabled": True,
                    },
                })
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            except OSError as exc:
                logger.exception("Could not save gate configuration")
                self._send_json({"error": f"Could not save gate config: {exc}"}, status=500)
            return

        self._send_json({"error": "Not Found", "path": path}, status=404)

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/logs/clear":
            self._send_json({"success": True, "message": "No local history store configured"})
            return
        self._send_json({"error": "Not Found", "path": path}, status=404)

    def log_message(self, format_string, *args):
        return


def _read_cpu_counters():
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            values = [int(v) for v in f.readline().split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, ValueError, IndexError):
        return None


_cpu_previous = _read_cpu_counters()


def _cpu_usage():
    global _cpu_previous
    current = _read_cpu_counters()
    previous = _cpu_previous
    _cpu_previous = current
    if current is None or previous is None:
        return 0
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    return round(max(0, min(100, 100 * (1 - idle_delta / total_delta))), 1) if total_delta > 0 else 0


def _gpu_usage():
    """Read Jetson Nano GPU load (0-1000) as a percentage (0-100)."""
    paths = (
        "/sys/devices/gpu.0/load",
        "/sys/devices/57000000.gpu/load",
    )

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                load = int(f.read().strip())
            if 0 <= load <= 1000:
                return round(load / 10.0, 1)
        except (OSError, ValueError):
            continue

    return None


def _memory_usage():
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, rest = line.split(":", 1)
                values[key] = int(rest.strip().split()[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        return round(100 * (total - available) / total, 1) if total else 0
    except (OSError, ValueError):
        return 0


def _temperatures():
    values = []
    try:
        names = os.listdir("/sys/class/thermal")
    except OSError:
        names = []
    for name in names:
        if not name.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join("/sys/class/thermal", name, "temp"), "r", encoding="utf-8") as f:
                value = float(f.read().strip())
            if value > 1000:
                value /= 1000
            if 0 <= value <= 150:
                values.append(value)
        except (OSError, ValueError):
            pass
    return values


def _network_counters():
    rx = tx = 0
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.readlines()[2:]
        for line in lines:
            if ":" not in line:
                continue
            iface, data = line.split(":", 1)
            iface = iface.strip()
            if iface == "lo" or iface.startswith(("docker", "veth", "br-")):
                continue
            fields = data.split()
            if len(fields) >= 9:
                rx += int(fields[0])
                tx += int(fields[8])
    except (OSError, ValueError):
        pass
    return rx, tx


def get_realtime_payload():
    global network_previous, network_speeds, network_last_update
    now = time.time()
    rx, tx = _network_counters()
    elapsed = max(0.001, now - network_last_update)
    if network_previous is not None:
        network_speeds = {
            "rx_speed": max(0, (rx - network_previous[0]) / elapsed),
            "tx_speed": max(0, (tx - network_previous[1]) / elapsed),
        }
    network_previous = (rx, tx)
    network_last_update = now

    temps = _temperatures()
    cpu_temp = round(temps[0], 1) if temps else 0
    gpu_temp = round(temps[1], 1) if len(temps) > 1 else cpu_temp
    chip_temp = round(max(temps), 1) if temps else cpu_temp
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            load_avg = [round(float(v), 2) for v in f.read().split()[:3]]
    except (OSError, ValueError):
        load_avg = []
    try:
        disk = shutil.disk_usage("/")
        disk_total = round(disk.total / (1024 ** 3), 1)
        disk_used = round(disk.used / (1024 ** 3), 1)
        disk_free = round(disk.free / (1024 ** 3), 1)
    except OSError:
        disk_total = None
        disk_used = None
        disk_free = None

    # Read only lightweight public fields from the latest AI detections.
    # Masks, boxes and other detector internals must not be JSON-encoded.
    with detection_lock:
        pigs = []
        for detection in latest_detection_results:
            raw_id = detection.get('track_id')
            try:
                track_id = int(raw_id) if raw_id is not None and not isinstance(raw_id, bool) else None
            except (TypeError, ValueError, OverflowError):
                track_id = None

            raw_weight = detection.get('weight')
            try:
                weight = float(raw_weight) if raw_weight is not None else None
            except (TypeError, ValueError, OverflowError):
                weight = None
            weight_kg = round(weight, 2) if weight is not None and math.isfinite(weight) and weight > 0 else None
            pigs.append({'track_id': track_id, 'weight_kg': weight_kg})

    with state_lock:
        count = int(latest_pig_count)
        in_count = int(latest_in_count)
        out_count = int(latest_out_count)
        net_count = int(latest_net_count)
        fps = round(float(latest_fps), 1)

    return {
        "fps": fps,
        "pig_counts": {"total": count, "in": in_count, "out": out_count, "net": net_count},
        "pigs": pigs,
        "hardware": {
            "cpu_usage": _cpu_usage(),
            "gpu_usage": _gpu_usage(),
            "ram_usage": _memory_usage(),
            "cpu_temp": cpu_temp,
            "gpu_temp": gpu_temp,
            "chip_temp": chip_temp,
            "load_avg": load_avg,
            "disk_total": disk_total,
            "disk_used": disk_used,
            "disk_free": disk_free,
        },
        "network": {
            "rx_speed": round(network_speeds["rx_speed"], 1),
            "tx_speed": round(network_speeds["tx_speed"], 1),
            "rx_bytes": rx,
            "tx_bytes": tx,
        },
    }


def start_stream_server():
    """
    Start the MJPEG HTTP server in a background thread.
    """

    server = ThreadingHTTPServer(
        (STREAM_HOST, STREAM_PORT),
        StreamHandler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    server_thread.start()

    logger.info(
        "Stream server started at http://0.0.0.0:%s | API=%s | module=%s | gate_config=%s",
        STREAM_PORT,
        API_VERSION,
        Path(__file__).resolve(),
        GATE_CONFIG_FILE,
    )

    return server


# ======================================================
# Asynchronous NAS API Upload
# ======================================================

class ApiUploadWorker:
    """Save weights and line-crossings in order, outside the AI processing loop."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.pending = queue.Queue(maxsize=256)
        self.stop_requested = threading.Event()
        self.last_weight_id = {}  # Track ID -> last saved weight estimation ID
        self.thread = threading.Thread(
            target=self._run, name="nas-api-uploader", daemon=True
        )
        self.thread.start()

    def submit(self, kind, **payload):
        """Never wait for network traffic on the AI thread."""
        try:
            self.pending.put_nowait((kind, payload))
            return True
        except queue.Full:
            logger.error("NAS API upload queue full; %s record not queued: %s", kind, payload)
            return False

    def _run(self):
        while not self.stop_requested.is_set() or not self.pending.empty():
            try:
                kind, payload = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                tid = payload['track_id']
                if kind == 'weight':
                    saved = send_weight(
                        session_id=self.session_id,
                        track_id=tid,
                        weight_kg=payload['weight_kg'],
                        frame_number=payload['frame_number'],
                    )
                    if saved and saved.get('estimation_id') is not None:
                        self.last_weight_id[tid] = saved['estimation_id']
                    else:
                        logger.warning('Weight upload failed: track=%s', tid)
                elif kind == 'crossing':
                    saved = send_counting_event(
                        session_id=self.session_id,
                        track_id=tid,
                        direction=payload['direction'],
                        frame_number=payload['frame_number'],
                        weight_estimation_id=self.last_weight_id.get(tid),
                    )
                    if saved is None:
                        logger.warning('Counting upload failed: track=%s', tid)
            except Exception:
                logger.exception('Unexpected NAS API upload error')
            finally:
                self.pending.task_done()

    def close(self):
        """Allow queued data to flush before ending the processing session."""
        self.stop_requested.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            logger.error('NAS API uploader still working; session left open to avoid closing early')
            return
        result = end_session(self.session_id)
        if result is None:
            logger.error('Could not close NAS processing session %s', self.session_id)
        else:
            logger.info('NAS processing session ended: %s', self.session_id)


# ======================================================
# Main Processing Loop
# ======================================================

def capture_camera_frames(camera):
    """Read frames and reconnect the USB camera if disconnected."""
    global latest_camera_frame, latest_camera_seq, latest_frame

    consecutive_failures = 0

    while running:
        try:
            success, frame = camera.read()
        except cv2.error:
            logger.exception("Camera read error")
            success, frame = False, None

        if not success or frame is None:
            consecutive_failures += 1

            if consecutive_failures < 5:
                time.sleep(0.1)
                continue

            logger.warning(
                "Camera disconnected. Attempting automatic reconnection..."
            )

            camera.release()

            # Clear cached frames so disconnected video is not reused.
            with camera_frame_lock:
                latest_camera_frame = None

            with frame_lock:
                latest_frame = None

            # Reopen the SAME VideoCapture object.
            # cleanup() can therefore release the latest camera correctly.
            reopened = open_camera(
                stop_on_shutdown=True,
                camera=camera,
            )

            if reopened is None:
                logger.info("Camera reconnection cancelled during shutdown")
                return

            consecutive_failures = 0
            logger.info("Camera reconnected successfully")
            continue

        consecutive_failures = 0

        # Keep only the newest frame.
        with camera_frame_lock:
            latest_camera_frame = frame
            latest_camera_seq += 1


def render_stream_frames(counter, recorder):
    """Render and publish every available camera frame using the latest AI results."""
    global latest_fps
    last_seq = -1
    fps_started = time.time()
    rendered_frames = 0

    while running:
        with camera_frame_lock:
            frame = latest_camera_frame.copy() if latest_camera_frame is not None else None
            seq = latest_camera_seq

        if frame is None or seq == last_seq:
            time.sleep(0.003)
            continue

        with detection_lock:
            results = list(latest_detection_results)
            track_info = dict(latest_track_info)
            predicted_boxes = list(latest_predicted_boxes)

        output = draw_results(
            frame,
            results,
            track_info=track_info,
            counter=counter,
            predicted_boxes=predicted_boxes,
        )
        publish_frame(output)
        # Save the same annotated frame that users see in the MJPEG stream.
        # submit() is non-blocking; disk I/O runs in the recorder worker.
        recorder.submit(output)
        last_seq = seq
        rendered_frames += 1
        elapsed = time.time() - fps_started
        if elapsed >= 1.0:
            with state_lock:
                latest_fps = rendered_frames / elapsed
            rendered_frames = 0
            fps_started = time.time()


def run_camera_loop(camera, detector):
    """Run AI on sampled latest frames while camera capture and MJPEG rendering stay independent."""
    global latest_pig_count, latest_in_count, latest_out_count, latest_net_count
    global latest_fps, latest_processing_ms, active_counter, running
    global latest_predicted_boxes

    counter = LineCrossingCounter()
    active_counter = counter
    latest_seq_processed = -1
    last_weight_queued_at = {}  # Track ID -> AI frame index

    # The NAS may be temporarily unavailable. Never prevent the AI from running.
    api_worker = None
    session = create_session(device_id=API_DEVICE_ID, source_type="LIVE")
    if session and session.get('session_id'):
        api_worker = ApiUploadWorker(session['session_id'])
        logger.info('NAS processing session started: %s', session['session_id'])
    else:
        logger.warning('NAS session unavailable; AI will run without database uploads')
    # ==================================================
    # Start Local Segmented MP4 Recording
    # ==================================================

    # Upload in a separate worker: recording and AI never wait for the NAS.
    # If session creation failed, keep MP4s locally without attempting upload.
    video_uploader = VideoUploadWorker(
        output_dir=RECORDING_DIRECTORY,
        api_base_url=API_BASE_URL,
        session_id=session.get('session_id') if session else None,
        delete_after_upload=True,
    )
    video_uploader.start()

    recorder = SegmentedVideoRecorder(
        output_dir=RECORDING_DIRECTORY,
        fps=RECORDING_FPS,
        segment_seconds=RECORDING_SEGMENT_SECONDS,
        on_segment_ready=video_uploader.queue_file,
    )
    recorder.start()

    capture_thread = threading.Thread(
        target=capture_camera_frames,
        args=(camera,),
        name="camera-capture",
        daemon=True,
    )
    render_thread = threading.Thread(
        target=render_stream_frames,
        args=(counter, recorder),
        name="camera-stream-render",
        daemon=True,
    )

    capture_thread.start()
    render_thread.start()

    try:
        while running:
            with camera_frame_lock:
                frame = latest_camera_frame.copy() if latest_camera_frame is not None else None
                seq = latest_camera_seq

            if frame is None or seq == latest_seq_processed:
                time.sleep(0.003)
                continue

            latest_seq_processed = seq

            # PROCESS_EVERY=1: no intentional frame modulo skipping.
            # Capture keeps the most recent image to avoid an inference backlog.

            start_time = time.time()
            results = process_frame(frame, detector)
            track_info, count_events = counter.update(results)

            if api_worker is not None:
                # Upload a first weight for each track and refresh occasionally.
                # At a line crossing, upload the current weight before its IN/OUT
                # event so the database event can reference that weight row.
                crossing_ids = {int(event['track_id']) for event in count_events}
                for detection in results:
                    tid = detection.get('track_id')
                    weight = detection.get('weight')
                    if tid is None or weight is None:
                        continue
                    try:
                        tid = int(tid)
                        weight = float(weight)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(weight) or weight <= 0:
                        continue
                    last_frame = last_weight_queued_at.get(tid)
                    due = (last_frame is None or
                           counter.frame_idx - last_frame >= WEIGHT_UPLOAD_INTERVAL_FRAMES)
                    if tid in crossing_ids or due:
                        accepted = api_worker.submit(
                            'weight', track_id=tid, weight_kg=weight,
                            frame_number=counter.frame_idx,
                        )
                        if accepted:
                            last_weight_queued_at[tid] = counter.frame_idx

                for event in count_events:
                    api_worker.submit(
                        'crossing', track_id=int(event['track_id']),
                        direction=event['direction'], frame_number=counter.frame_idx,
                    )

            for event in count_events:
                logger.info(
                    "Pig crossing: %(direction)s track=%(track_id)s IN=%(in_count)s OUT=%(out_count)s",
                    event,
                )

            processing_time = (time.time() - start_time) * 1000
            with detection_lock:
                latest_detection_results[:] = results
                latest_track_info.clear()
                latest_track_info.update(track_info)
                latest_predicted_boxes[:] = counter.predicted_boxes

            with state_lock:
                latest_pig_count = len(results)
                latest_in_count = counter.in_count
                latest_out_count = counter.out_count
                latest_net_count = counter.net_count
                latest_processing_ms = processing_time

            logger.debug(
                'Pig count: %d | AI processing: %.2f ms',
                len(results), processing_time,
            )
    finally:
        running = False
        capture_thread.join(timeout=2)
        render_thread.join(timeout=2)
        recorder.stop()
        video_uploader.stop(wait_seconds=20)
        if api_worker is not None:
            api_worker.close()


# ======================================================
# Release Resources
# ======================================================

def cleanup(camera, detector, server):
    """
    Release camera, AI model, HTTP server,
    and OpenCV windows.
    """

    global running

    running = False

    logger.info("Cleaning up resources...")

    if server is not None:
        server.shutdown()
        server.server_close()

    if camera is not None:
        camera.release()

    if detector is not None:
        detector.close()

    # No GUI windows are opened in headless mode.

    logger.info("Resources released")


# ======================================================
# Main
# ======================================================

def main():
    """
    Initialize the camera and AI model,
    then start real-time pig detection.
    """

    # Enable on-demand thread diagnostics.
    import faulthandler
    import signal
    import sys

    faulthandler.register(
        signal.SIGUSR1,
        file=sys.stderr,
        all_threads=True,
    )
    logger.info("Thread diagnostics enabled (SIGUSR1)")

    camera = None
    detector = None
    server = None

    try:
        # ----------------------------------------------
        # Initialize Camera
        # ----------------------------------------------

        camera = open_camera()

        # ----------------------------------------------
        # Load AI Model
        # ----------------------------------------------

        detector = load_model()

        # ----------------------------------------------
        # Start MJPEG Server
        # ----------------------------------------------

        server = start_stream_server()

        # ----------------------------------------------
        # Start Real-Time Processing
        # ----------------------------------------------

        run_camera_loop(
            camera,
            detector,
        )

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")

    except Exception:
        logger.exception("Vision Swine stopped due to an error")
        raise

    finally:
        cleanup(
            camera,
            detector,
            server,
        )


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()