# ======================================================
# Vision Swine V2 - Jetson Nano API Client
# ======================================================

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Backend on the NAS (reachable from this Jetson over Tailscale).
API_BASE_URL = "http://100.123.171.62:8002"
API_TIMEOUT = 10


# ======================================================
# Shared HTTP Helper
# ======================================================

def _request(method, path, payload=None):
    """Return the decoded API response, or None if the request fails.

    Do not automatically retry writes: a timeout may occur after the NAS
    has already saved the record, and retrying could create duplicates.
    """
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        try:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            print("[API] Invalid payload:", exc)
            return None
        headers["Content-Type"] = "application/json"

    request = Request(
        url=API_BASE_URL.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=API_TIMEOUT) as response:
            content = response.read().decode("utf-8")
            return json.loads(content) if content else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print("[API] {} {}: HTTP {} - {}".format(
            method, path, exc.code, detail
        ))
    except (URLError, TimeoutError, OSError) as exc:
        print("[API] {} {}: Connection error - {}".format(
            method, path, exc
        ))
    except (ValueError, UnicodeError) as exc:
        print("[API] {} {}: Invalid response - {}".format(
            method, path, exc
        ))

    return None


# ======================================================
# Processing Sessions
# ======================================================

def get_sessions():
    """Read existing sessions without changing database records."""
    return _request("GET", "/sessions")


def create_session(device_id=1, source_type="LIVE", camera_id=None):
    """Create one session at the beginning of a processing run."""
    payload = {
        "device_id": device_id,
        "source_type": source_type,
    }
    if camera_id is not None:
        payload["camera_id"] = camera_id
    return _request("POST", "/sessions", payload)


def end_session(session_id):
    """Close a session when the AI processing run has stopped."""
    return _request("PATCH", "/sessions/{}/end".format(session_id))


# ======================================================
# Weight Estimations
# ======================================================

def send_weight(session_id, track_id, weight_kg, frame_number):
    """Save a weight estimate and return its estimation_id."""
    return _request("POST", "/weights", {
        "session_id": str(session_id),
        "track_id": int(track_id),
        "weight_kg": round(float(weight_kg), 2),
        "frame_number": int(frame_number),
    })


# ======================================================
# IN / OUT Counting Events
# ======================================================

def send_counting_event(
    session_id,
    track_id,
    direction,
    frame_number=None,
    weight_estimation_id=None,
    event_time=None,
):
    """Save a real line crossing, optionally linked to a saved weight."""
    payload = {
        "session_id": str(session_id),
        "track_id": int(track_id),
        "direction": str(direction).upper(),
    }
    if frame_number is not None:
        payload["frame_number"] = int(frame_number)
    if weight_estimation_id is not None:
        payload["weight_estimation_id"] = int(weight_estimation_id)
    if event_time is not None:
        payload["event_time"] = event_time

    return _request("POST", "/counting-events", payload)


# ======================================================
# Legacy API Compatibility
# ======================================================

def send_event(*args, **kwargs):
    """Old video_processor.py imports this; /events no longer exists."""
    print("[API] Legacy send_event() is disabled: /events was removed in V2.")
    return None
