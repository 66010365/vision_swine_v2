
# ======================================================
# Import Libraries
# ======================================================

import cv2
import numpy as np
import time
import math


# ======================================================
# Import Project Modules
# ======================================================

from .config import (
    INPUT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    PROCESS_EVERY,
    MAX_SECONDS,
)

from .detector import PigDetector
from .features import extract_features
from .weight_predictor import predict_weight
from .api_client import send_event


# ======================================================
# API Configuration
# ======================================================

# ID of the camera registered in the backend database.
# Current test camera ID in PostgreSQL: 1
CAMERA_ID = 1


# ======================================================
# Draw AI Results
# ======================================================

def draw_results(frame, results):
    """
    Draw pig masks, bounding boxes, confidence,
    and estimated weights on the frame.
    """

    output = frame.copy()

    # Create a combined mask for all detected pigs

    combined_mask = np.zeros(
        frame.shape[:2],
        dtype=bool
    )

    for result in results:

        mask = np.asarray(result["mask"]) > 0

        if mask.shape == combined_mask.shape:
            combined_mask |= mask

    # Draw mask overlay

    overlay = output.copy()
    overlay[combined_mask] = (0, 255, 0)

    output = cv2.addWeighted(
        output,
        0.75,
        overlay,
        0.25,
        0
    )

    # Display total pig count

    cv2.putText(
        output,
        "Pigs detected: {}".format(len(results)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Draw each pig's bounding box and information

    for index, result in enumerate(results, start=1):

        x1, y1, x2, y2 = result["box"]
        confidence = result["confidence"]
        weight = result.get("weight")

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if weight is not None:

            label = (
                "Pig {} | {:.2f} | {:.2f} kg"
                .format(index, confidence, weight)
            )

        else:

            label = (
                "Pig {} | {:.2f} | Weight N/A"
                .format(index, confidence)
            )

        label_y = max(25, y1 - 10)

        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


# ======================================================
# Process One AI Inference Frame
# ======================================================

def process_inference_frame(frame, detector):
    """
    Detect pigs, extract features, and estimate weights.

    Returns:
        current_results: List of pig detection results
        processing_time_ms: AI processing duration
    """

    # Start processing timer

    start_time = time.perf_counter()

    # Detect pigs in the current frame

    detections = detector.detect(frame)

    current_results = []

    print(
        "\nFrame inference: {} detections".format(
            len(detections)
        )
    )

    # Process each detected pig

    for pig_index, detection in enumerate(
        detections,
        start=1
    ):

        # Extract features from pig mask

        features = extract_features(
            detection["mask"]
        )

        if features is None:

            detection["weight"] = None
            current_results.append(detection)

            print(
                "Pig {}: feature extraction failed"
                .format(pig_index)
            )

            continue

        detection["features"] = features

        # Estimate pig weight

        try:

            # Predict weight and validate the returned value.
            predicted_weight = predict_weight(features)

            if predicted_weight is None:
                detection["weight"] = None

                print(
                    "Pig {} | Confidence: {:.4f} | "
                    "Weight prediction unavailable (None)"
                    .format(
                        pig_index,
                        detection["confidence"]
                    )
                )

            else:
                # Convert numeric-like values (including NumPy scalars)
                # to a Python float before formatting or drawing.
                weight = float(predicted_weight)

                # Reject NaN and infinity because they are not valid weights.
                if not math.isfinite(weight):
                    detection["weight"] = None

                    print(
                        "Pig {} | Confidence: {:.4f} | "
                        "Weight prediction invalid: {}"
                        .format(
                            pig_index,
                            detection["confidence"],
                            predicted_weight
                        )
                    )
                else:
                    detection["weight"] = weight

                    print(
                        "Pig {} | Confidence: {:.4f} | "
                        "Estimated weight: {:.2f} kg"
                        .format(
                            pig_index,
                            detection["confidence"],
                            weight
                        )
                    )

        except (TypeError, ValueError, OverflowError) as exc:

            detection["weight"] = None

            print(
                "Pig {}: invalid weight prediction: {}"
                .format(pig_index, exc)
            )

        except Exception as exc:

            detection["weight"] = None

            print(
                "Pig {}: weight prediction failed: {}"
                .format(pig_index, exc)
            )

        current_results.append(detection)

    # Calculate AI processing duration

    processing_time_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    return current_results, processing_time_ms


# ======================================================
# Send Results to Backend API
# ======================================================

def upload_results(current_results, processing_time_ms):
    """
    Send the current pig count to the backend.

    API errors should not stop video processing.
    """

    pig_count = len(current_results)

    try:

        send_event(
            camera_id=CAMERA_ID,
            pig_count=pig_count,
            processing_time_ms=processing_time_ms
        )

    except Exception as exc:

        print(
            "[API] Unexpected error; "
            "video processing will continue: {}"
            .format(exc)
        )


# ======================================================
# Main Video Processing
# ======================================================

def process_video():
    """
    Process a test video, estimate pig weights,
    save annotated output, and upload AI events.
    """

    # --------------------------------------------------
    # Prepare Input and Output
    # --------------------------------------------------

    video_path = INPUT_VIDEO_PATH
    output_path = OUTPUT_VIDEO_PATH

    if not video_path.is_file():

        raise FileNotFoundError(
            "Input video not found: {}".format(
                video_path
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Open Input Video
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open video: {}".format(
                video_path
            )
        )

    detector = None
    writer = None

    try:

        # --------------------------------------------------
        # Read Video Information
        # --------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        max_frames = int(
            fps * MAX_SECONDS
        )

        # --------------------------------------------------
        # Create Output Video Writer
        # --------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Cannot create output video: {}".format(
                    output_path
                )
            )

        # --------------------------------------------------
        # Print Processing Information
        # --------------------------------------------------

        print("=" * 55)
        print("VISION SWINE - VIDEO PROCESSOR")
        print("Input:", video_path)
        print("Output:", output_path)
        print(
            "Video size:",
            frame_width,
            "x",
            frame_height
        )
        print("FPS:", fps)
        print("Source frames:", total_frames)
        print(
            "Processing limit:",
            MAX_SECONDS,
            "seconds"
        )
        print(
            "AI runs every",
            PROCESS_EVERY,
            "frames"
        )
        print("Camera ID:", CAMERA_ID)
        print("Backend API upload: enabled")
        print("=" * 55)

        # --------------------------------------------------
        # Initialize AI Detector
        # --------------------------------------------------

        detector = PigDetector()

        frame_index = 0
        inference_count = 0
        last_results = []

        # --------------------------------------------------
        # Main Frame Loop
        # --------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret or frame_index >= max_frames:
                break

            # ----------------------------------------------
            # Run AI at Configured Intervals
            # ----------------------------------------------

            if frame_index % PROCESS_EVERY == 0:

                inference_count += 1

                print(
                    "\nFrame {}"
                    .format(frame_index)
                )

                # Run detection and weight estimation

                current_results, processing_time_ms = (
                    process_inference_frame(
                        frame,
                        detector
                    )
                )

                # Save the latest AI results

                last_results = current_results

                # ------------------------------------------
                # Upload Results to Backend
                # ------------------------------------------

                upload_results(
                    current_results,
                    processing_time_ms
                )

            # ----------------------------------------------
            # Draw Results and Save Video Frame
            # ----------------------------------------------

            output_frame = draw_results(
                frame,
                last_results
            )

            writer.write(output_frame)

            frame_index += 1

        # --------------------------------------------------
        # Print Summary
        # --------------------------------------------------

        print("\n" + "=" * 55)
        print("VIDEO PROCESSING COMPLETED")
        print("Frames written:", frame_index)
        print("AI inference frames:", inference_count)
        print("Output:", output_path)
        print("=" * 55)

    finally:

        # --------------------------------------------------
        # Release Resources
        # --------------------------------------------------

        if writer is not None:
            writer.release()

        cap.release()

        if detector is not None:
            detector.close()


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    process_video()
