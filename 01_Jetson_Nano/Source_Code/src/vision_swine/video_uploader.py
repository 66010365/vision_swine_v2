"""Background MP4 uploader for Vision Swine V2.

Completed MP4 segments are tracked with durable sidecar manifests. The NAS
upload runs in its own thread; failed uploads remain on Jetson for retry on
next run. Local video is removed only after NAS metadata and HTTP Range access are
verified. Chrome decoding still needs a separate browser check. A processing session ID must exist for each video upload.
"""

# ======================================================
# Import Libraries
# ======================================================

import http.client
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


# ======================================================
# Configuration
# ======================================================

LOGGER = logging.getLogger(__name__)
DEFAULT_API_TIMEOUT = 45
DEFAULT_RETRY_SECONDS = 30
MANIFEST_SUFFIX = ".upload.json"
H264_CACHE_DIR = ".h264_ready"
H264_ENCODER = "libx264"
H264_PRESET = "ultrafast"
H264_CRF = "25"
H264_THREADS = "1"  # Limit extra CPU load while AI is running.
H264_CONVERT_TIMEOUT = 900



# ======================================================
# NAS Multipart Upload
# ======================================================

def _upload_mp4(api_base_url, session_id, video_path, timeout):
    """Stream a multipart MP4 upload; never load the whole file into RAM."""
    url = urlsplit(api_base_url.rstrip("/") + "/videos")
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("Invalid NAS API URL")

    filename = video_path.name
    if not filename.lower().endswith(".mp4"):
        raise ValueError("Only finished MP4 files may be uploaded")

    boundary = "vision-swine-{}".format(uuid4().hex)
    prefix = (
        "--{b}\r\n"
        'Content-Disposition: form-data; name="session_id"\r\n\r\n'
        "{sid}\r\n"
        "--{b}\r\n"
        'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).format(b=boundary, sid=session_id, name=filename).encode("utf-8")
    suffix = "\r\n--{}--\r\n".format(boundary).encode("ascii")
    size = video_path.stat().st_size
    total_size = len(prefix) + size + len(suffix)

    connection_type = (
        http.client.HTTPSConnection if url.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(url.hostname, url.port, timeout=timeout)
    try:
        connection.putrequest("POST", url.path)
        connection.putheader("Content-Type", "multipart/form-data; boundary={}".format(boundary))
        connection.putheader("Content-Length", str(total_size))
        connection.putheader("Accept", "application/json")
        connection.endheaders()
        connection.send(prefix)
        with video_path.open("rb") as source:
            while True:
                chunk = source.read(256 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        connection.send(suffix)

        response = connection.getresponse()
        payload = response.read()
        if response.status != 201:
            raise RuntimeError("NAS upload HTTP {}: {}".format(
                response.status, payload[:250].decode("utf-8", errors="replace")
            ))
        result = json.loads(payload.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("NAS upload returned invalid JSON")
        return result
    finally:
        connection.close()


def _get_video_list(api_base_url, timeout):
    request = Request(api_base_url.rstrip("/") + "/videos", headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, list):
        raise ValueError("NAS video list response is not a list")
    return result


def _find_ready_video(videos, manifest, video_path):
    """Match NAS records to a segment by session, original name and size."""
    expected_size = video_path.stat().st_size
    for item in videos:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("session_id")) == str(manifest["session_id"])
            and item.get("file_name") == video_path.name
            and item.get("file_size_bytes") == expected_size
            and item.get("status") == "READY"
            and item.get("video_id") is not None
        ):
            return item
    return None


def _verify_video_playback(api_base_url, video_id, timeout):
    """Check that NAS supports reading the saved MP4 (not browser decoding)."""
    request = Request(
        "{}/videos/{}/file".format(api_base_url.rstrip("/"), video_id),
        headers={"Range": "bytes=0-0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return (
            response.status == 206
            and response.headers.get_content_type() == "video/mp4"
            and len(response.read(1)) == 1
        )


# ======================================================
# Browser-Compatible H.264 Conversion
# ======================================================

def _probe_video(video_path):
    """Get video properties from ffprobe; reject unreadable MP4 files."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt:format=duration",
            "-of", "json", str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=60,
        check=True,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams") or []
    duration = float((info.get("format") or {}).get("duration") or 0)
    if not streams or duration <= 0:
        raise ValueError("Video has no valid video stream: {}".format(video_path))
    return streams[0], duration


def _is_browser_h264(video_path):
    stream, duration = _probe_video(video_path)
    return stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p", duration


def _prepare_h264_video(video_path):
    """Convert to H.264 using a cache; preserve original until NAS verifies.

    Temp files remain local and never get upload manifests. A completed cached
    conversion is reused after network failures or application restarts.
    """
    original_ok, original_duration = _is_browser_h264(video_path)
    if original_ok:
        return video_path

    cache_dir = video_path.parent / H264_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    converted_path = cache_dir / video_path.name
    temp_path = cache_dir / (video_path.stem + ".transcoding.mp4")

    if converted_path.is_file():
        try:
            cache_ok, cached_duration = _is_browser_h264(converted_path)
            if cache_ok and abs(cached_duration - original_duration) <= 1.0:
                LOGGER.info("Reusing cached H.264: %s", converted_path.name)
                return converted_path
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
        converted_path.unlink()

    # Remove a temp file left by an interrupted previous transcoding attempt.
    if temp_path.exists():
        temp_path.unlink()

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-map", "0:v:0",
        "-c:v", H264_ENCODER, "-preset", H264_PRESET,
        "-crf", H264_CRF, "-pix_fmt", "yuv420p",
        "-threads", H264_THREADS,
        "-an", "-movflags", "+faststart", "-f", "mp4",
        str(temp_path),
    ]
    LOGGER.info("Converting to H.264: %s", video_path.name)
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=H264_CONVERT_TIMEOUT,
        )
        converted_ok, converted_duration = _is_browser_h264(temp_path)
        if not converted_ok or abs(converted_duration - original_duration) > 1.0:
            raise ValueError("Invalid H.264 conversion: {}".format(video_path.name))
        os.replace(str(temp_path), str(converted_path))
        LOGGER.info("H.264 conversion ready: %s", converted_path)
        return converted_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ======================================================
# Durable Background Upload Worker
# ======================================================

class VideoUploadWorker:
    """Upload recorded MP4s to NAS without blocking the AI/recorder.

    queue_file() is called by SegmentedVideoRecorder.on_segment_ready.
    Only files with a durable *.upload.json sidecar are auto-uploaded.
    Such sidecars survive restart and prevent losing the session association.
    """

    def __init__(
        self,
        output_dir,
        api_base_url,
        session_id,
        timeout=DEFAULT_API_TIMEOUT,
        retry_seconds=DEFAULT_RETRY_SECONDS,
        delete_after_upload=True,
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.api_base_url = str(api_base_url).rstrip("/")
        self.session_id = str(session_id) if session_id else None
        self.timeout = float(timeout)
        self.retry_seconds = float(retry_seconds)
        self.delete_after_upload = bool(delete_after_upload)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread = None
        self._manifest_lock = threading.Lock()

    # ==================================================
    # Start and Stop Worker
    # ==================================================

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Video uploader already started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="vision-swine-video-upload",
            daemon=True,
        )
        self._thread.start()
        self._wake_event.set()  # Retry pending segments left from previous runs.
        LOGGER.info("NAS video uploader started: %s", self.api_base_url)

    def stop(self, wait_seconds=20):
        """Stop without risking an indefinite wait on network outages."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=wait_seconds)
        if self._thread.is_alive():
            LOGGER.warning("NAS upload still in progress; local MP4 remains safe for retry")
        else:
            LOGGER.info("NAS video uploader stopped")
        self._thread = None

    # ==================================================
    # Queue Completed Segments
    # ==================================================

    def queue_file(self, path):
        """Persist the session association; return immediately after local I/O."""
        video_path = Path(path)
        if not video_path.is_file() or video_path.name.endswith(".part.mp4"):
            LOGGER.error("Cannot queue unfinished or missing video: %s", video_path)
            return False
        if not self.session_id:
            LOGGER.warning("No NAS session; keeping video on Jetson: %s", video_path)
            return False

        manifest_path = Path(str(video_path) + MANIFEST_SUFFIX)
        content = {"session_id": self.session_id, "filename": video_path.name}
        temp_path = Path(str(manifest_path) + ".tmp")
        try:
            with self._manifest_lock:
                with temp_path.open("w", encoding="utf-8") as target:
                    json.dump(content, target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(str(temp_path), str(manifest_path))
        except OSError:
            LOGGER.exception("Cannot save upload manifest: %s", manifest_path)
            return False

        self._wake_event.set()
        LOGGER.info("Queued MP4 for NAS: %s", video_path.name)
        return True

    # ==================================================
    # Upload One Completed Video
    # ==================================================

    def _process_manifest(self, manifest_path):
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
        video_path = Path(str(manifest_path)[:-len(MANIFEST_SUFFIX)])
        if not video_path.is_file():
            LOGGER.warning("Missing video for upload manifest: %s", video_path)
            return
        if not manifest.get("session_id") or manifest.get("filename") != video_path.name:
            LOGGER.error("Invalid video upload manifest: %s", manifest_path)
            return

        # Convert before uploading, without modifying the original recording.
        # All network requests run on the uploader's dedicated worker thread.
        upload_path = _prepare_h264_video(video_path)

        # First check for a previously completed upload (e.g., lost HTTP reply).
        # The current backend lists only the latest 100 videos.
        videos = _get_video_list(self.api_base_url, self.timeout)
        existing = _find_ready_video(videos, manifest, upload_path)
        if existing is not None:
            result = existing
            LOGGER.info("NAS already has video %s (id=%s)", upload_path.name, result["video_id"])
        else:
            result = _upload_mp4(
                self.api_base_url,
                manifest["session_id"],
                upload_path,
                self.timeout,
            )
            expected_size = upload_path.stat().st_size
            if (
                result.get("status") != "READY"
                or str(result.get("session_id")) != str(manifest["session_id"])
                or result.get("file_name") != upload_path.name
                or result.get("file_size_bytes") != expected_size
                or result.get("video_id") is None
            ):
                raise ValueError("NAS did not confirm matching READY video metadata")

        if not _verify_video_playback(self.api_base_url, result["video_id"], self.timeout):
            raise RuntimeError("NAS video read verification failed; keeping local video")

        if self.delete_after_upload:
            # Delete both only after server metadata and Range response are valid.
            if upload_path != video_path:
                upload_path.unlink()
            video_path.unlink()
            LOGGER.info("NAS verified: video_id=%s; removed local MP4 %s", result["video_id"], video_path.name)
        else:
            LOGGER.info("NAS verified: video_id=%s; keeping local MP4", result["video_id"])
        manifest_path.unlink()

    # ==================================================
    # Background Retry Loop
    # ==================================================

    def _run(self):
        while not self._stop_event.is_set():
            manifests = sorted(self.output_dir.glob("*.mp4" + MANIFEST_SUFFIX))
            failed = False
            for manifest_path in manifests:
                if self._stop_event.is_set():
                    break
                try:
                    self._process_manifest(manifest_path)
                except (OSError, ValueError, RuntimeError, HTTPError, URLError, json.JSONDecodeError):
                    LOGGER.exception("NAS upload pending for retry: %s", manifest_path.name)
                    failed = True
                    break
                except Exception:
                    LOGGER.exception("Unexpected upload problem: %s", manifest_path.name)
                    failed = True
                    break

            if self._stop_event.is_set():
                break
            self._wake_event.clear()
            if failed:
                # Don't hammer the NAS. Files remain on disk for later retries.
                self._stop_event.wait(self.retry_seconds)
            else:
                self._wake_event.wait(self.retry_seconds)
