"""Record Vision Swine frames as consecutive wall-clock timed MP4 segments.

The recorder outputs a constant-frame-rate MP4 by repeating the latest AI
frame when new frames arrive more slowly than the configured video FPS.
This preserves the elapsed wall-clock duration in the saved recording,
without running the AI model again.
"""

# ======================================================
# Import Libraries
# ======================================================

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


# ======================================================
# Logging and Default Configuration
# ======================================================

LOGGER = logging.getLogger(__name__)

DEFAULT_FPS = 15.0
DEFAULT_SEGMENT_SECONDS = 180
DEFAULT_QUEUE_SIZE = 12
VIDEO_CODEC = "mp4v"
VIDEO_PREFIX = "vision_swine"


# ======================================================
# Segmented Video Recorder
# ======================================================

class SegmentedVideoRecorder:
    """Write submitted BGR frames in a background thread.

    Usage::

        recorder = SegmentedVideoRecorder("outputs/recordings", fps=15)
        recorder.start()
        recorder.submit(annotated_frame)
        recorder.stop()

    The optional on_segment_ready callback receives a completed MP4 path.
    It should enqueue uploads, not perform a slow network upload directly.
    """

    # ==================================================
    # Initialize Recorder
    # ==================================================

    def __init__(
        self,
        output_dir,
        fps=DEFAULT_FPS,
        segment_seconds=DEFAULT_SEGMENT_SECONDS,
        queue_size=DEFAULT_QUEUE_SIZE,
        on_segment_ready=None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")

        self.output_dir = Path(output_dir).expanduser()
        self.fps = float(fps)
        self.segment_seconds = float(segment_seconds)
        self.on_segment_ready = on_segment_ready

        self._frames = queue.Queue(maxsize=int(queue_size))
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._writer = None
        self._partial_path = None
        self._started_at = None
        self._frame_size = None
        self._segment_index = 0
        self._latest_frame = None
        self._next_frame_at = None
        self._stop_at = None

        self._dropped_frames = 0
        self._completed_files = []

    # ==================================================
    # Start and Stop Recording
    # ==================================================

    def start(self):
        """Start the recording worker without opening another camera."""
        if self._thread is not None:
            raise RuntimeError("Recorder is already started")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="vision-swine-recorder",
            daemon=False,
        )
        self._thread.start()

        LOGGER.info(
            "MP4 recorder started (segment length: %.0f seconds)",
            self.segment_seconds,
        )

    def stop(self):
        """Finish queued frames and finalize the last MP4 file."""
        if self._thread is None:
            return

        # Remember the requested stop time so the last recording does not
        # grow artificially while queued frames are being flushed.
        self._stop_at = time.monotonic()
        self._stop_event.set()
        self._thread.join()
        self._thread = None

        LOGGER.info(
            "MP4 recorder stopped; dropped frames: %d",
            self.dropped_frames,
        )

    # ==================================================
    # Receive Frames from the Main AI Program
    # ==================================================

    def submit(self, frame):
        """Queue one BGR frame without waiting for video disk writes."""
        if self._thread is None or self._stop_event.is_set():
            return False

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return False

        try:
            self._frames.put_nowait((time.monotonic(), frame.copy()))
            return True
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1
            return False

    # ==================================================
    # Recording Status
    # ==================================================

    @property
    def completed_files(self):
        """Return the paths of MP4 segments completed this run."""
        with self._lock:
            return list(self._completed_files)

    @property
    def dropped_frames(self):
        """Count frames skipped because the recording queue was full."""
        with self._lock:
            return self._dropped_frames

    # ==================================================
    # Open a New MP4 Segment
    # ==================================================

    def _open_segment(self, timestamp, frame):
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        self._started_at = timestamp
        self._segment_index += 1

        date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}_{}_{:04d}.part.mp4".format(
            VIDEO_PREFIX,
            date_string,
            self._segment_index,
        )
        self._partial_path = self.output_dir / filename

        codec = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(
            str(self._partial_path),
            codec,
            self.fps,
            self._frame_size,
        )

        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError(
                "Cannot create MP4 file: {}".format(self._partial_path)
            )

        LOGGER.info("Recording video: %s", self._partial_path)

    # ==================================================
    # Finalize a Completed MP4 Segment
    # ==================================================

    def _finalize_segment(self):
        if self._writer is None:
            return

        self._writer.release()
        self._writer = None

        partial_path = self._partial_path
        self._partial_path = None

        if (
            partial_path is None
            or not partial_path.is_file()
            or partial_path.stat().st_size == 0
        ):
            LOGGER.warning("MP4 segment is empty or missing: %s", partial_path)
            return

        final_path = partial_path.with_name(
            partial_path.name.replace(".part.mp4", ".mp4")
        )
        partial_path.rename(final_path)

        with self._lock:
            self._completed_files.append(final_path)

        LOGGER.info("MP4 segment ready: %s", final_path)

        if self.on_segment_ready is not None:
            try:
                self.on_segment_ready(final_path)
            except Exception:
                LOGGER.exception("Segment-ready callback failed")

    # ==================================================
    # Write Video Frames at Their Real-Time Positions
    # ==================================================

    def _write_until(self, target_timestamp):
        """Fill all output frame slots preceding the given monotonic time.

        A frame is written for every 1 / fps seconds of real time. If AI
        hasn't produced another frame yet, the most recent frame is reused.
        Every segment still covers a fixed wall-clock interval.
        """
        if self._writer is None or self._latest_frame is None:
            return

        frame_interval = 1.0 / self.fps

        while self._next_frame_at < target_timestamp:
            # Rotate *before* writing a frame assigned to the next segment.
            # This works even when no fresh AI frames arrive at the boundary.
            if self._next_frame_at >= self._started_at + self.segment_seconds - 1e-7:
                new_start = self._next_frame_at
                self._finalize_segment()
                self._open_segment(new_start, self._latest_frame)

            self._writer.write(self._latest_frame)
            self._next_frame_at += frame_interval

    # ==================================================
    # Background Recording Loop
    # ==================================================

    def _run(self):
        """Receive AI frames and record a steady wall-clock FPS timeline."""
        try:
            while True:
                try:
                    timestamp, frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    if self._writer is not None:
                        # Even when there is no new camera/AI frame, preserve
                        # elapsed time with the most recent rendered frame.
                        target = (
                            self._stop_at
                            if self._stop_event.is_set() and self._stop_at is not None
                            else time.monotonic()
                        )
                        self._write_until(target)

                    if self._stop_event.is_set() and self._frames.empty():
                        break
                    continue

                try:
                    if self._writer is None:
                        self._open_segment(timestamp, frame)
                        self._latest_frame = frame
                        self._next_frame_at = timestamp
                        continue

                    height, width = frame.shape[:2]
                    if (width, height) != self._frame_size:
                        LOGGER.warning(
                            "Skipping frame with changed resolution: %sx%s",
                            width,
                            height,
                        )
                        continue

                    # Fill the interval BEFORE the new frame arrived using
                    # the previous frame; after that, use the new frame.
                    self._write_until(timestamp)
                    self._latest_frame = frame
                finally:
                    self._frames.task_done()

            # Flush the final fraction of a segment on Ctrl+C / shutdown.
            if self._writer is not None and self._stop_at is not None:
                self._write_until(self._stop_at)

        except Exception:
            LOGGER.exception("MP4 recording failed")
        finally:
            self._finalize_segment()
