# ======================================================
# Import Libraries
# ======================================================

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


# ======================================================
# YAML Runtime Settings (config.py is NOT the detection source of truth)
# ======================================================

from pathlib import Path
from .runtime_config import load_runtime_config


# ======================================================
# TensorRT Configuration
# ======================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ======================================================
# Pig Detector
# ======================================================

class PigDetector:
    """Detect pigs and generate segmentation masks using TensorRT."""

    # ==================================================
    # Initialize TensorRT Model
    # ==================================================

    def __init__(self, *, engine_path=None, input_size=None,
                 conf_threshold=None, nms_threshold=None,
                 mask_threshold=None, max_detections=None):
        cfg = load_runtime_config()['model']
        self.engine_path = Path(engine_path or cfg['resolved_path'])
        self.input_size = int(input_size if input_size is not None else cfg.get('input_size', 960))
        self.conf_threshold = float(conf_threshold if conf_threshold is not None else cfg.get('confidence_threshold', 0.15))
        self.nms_threshold = float(nms_threshold if nms_threshold is not None else cfg.get('nms_threshold', 0.45))
        self.mask_threshold = float(mask_threshold if mask_threshold is not None else cfg.get('mask_threshold', 0.50))
        self.max_detections = int(max_detections if max_detections is not None else cfg.get('max_detections', 100))
        if self.input_size <= 0 or self.max_detections <= 0:
            raise ValueError('input_size and max_detections must be positive')
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {self.engine_path}"
            )

        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
        self.closed = False

        try:
            with open(self.engine_path, "rb") as file:
                engine_data = file.read()

            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(
                engine_data
            )

            if self.engine is None:
                raise RuntimeError("Could not load TensorRT engine")

            self.context = self.engine.create_execution_context()

            if self.context is None:
                raise RuntimeError(
                    "Could not create TensorRT execution context"
                )

            # Use the verified binding names from our engine.
            self.input_index = self.engine.get_binding_index("images")
            self.output1_index = self.engine.get_binding_index("output1")
            self.output0_index = self.engine.get_binding_index("output0")

            if min(
                self.input_index,
                self.output1_index,
                self.output0_index,
            ) < 0:
                raise RuntimeError("Expected TensorRT binding not found")

            input_shape = tuple(
                self.engine.get_binding_shape(self.input_index)
            )

            if input_shape != (1, 3, self.input_size, self.input_size):
                raise RuntimeError(
                    f"Unexpected input shape: {input_shape}"
                )

            output0_shape = tuple(
                self.engine.get_binding_shape(self.output0_index)
            )
            output1_shape = tuple(
                self.engine.get_binding_shape(self.output1_index)
            )

            if output0_shape != (1, 37, 18900):
                raise RuntimeError(
                    f"Unexpected output0 shape: {output0_shape}"
                )

            if output1_shape != (1, 32, 240, 240):
                raise RuntimeError(
                    f"Unexpected output1 shape: {output1_shape}"
                )

            self.stream = cuda.Stream()

            self.d_input = cuda.mem_alloc(
                int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
            )
            self.d_output0 = cuda.mem_alloc(
                int(np.prod(output0_shape)) * np.dtype(np.float32).itemsize
            )
            self.d_output1 = cuda.mem_alloc(
                int(np.prod(output1_shape)) * np.dtype(np.float32).itemsize
            )

            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.input_index] = int(self.d_input)
            self.bindings[self.output0_index] = int(self.d_output0)
            self.bindings[self.output1_index] = int(self.d_output1)

            print("TensorRT engine loaded successfully")
            print("GPU:", cuda.Device(0).name())
            print("Input:", input_shape)
            print("Output0:", output0_shape)
            print("Output1:", output1_shape)

        except Exception:
            self.cuda_context.pop()
            self.cuda_context.detach()
            raise

    # ==================================================
    # Preprocess Image
    # ==================================================

    def preprocess(self, frame):
        """Convert an OpenCV BGR frame into model input format."""
        image = cv2.resize(
            frame,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return np.ascontiguousarray(image)

    # ==================================================
    # Sigmoid Function
    # ==================================================

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -50, 50)
        return 1.0 / (1.0 + np.exp(-values))

    # ==================================================
    # Run Detection
    # ==================================================

    def detect(self, frame):
        """
        Detect pigs in one frame.

        Returns a list of dictionaries containing:
        box: [x1, y1, x2, y2]
        confidence: detection confidence
        mask: uint8 segmentation mask, 0 or 255
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        input_data = self.preprocess(frame)

        host_output0 = np.empty(
            (1, 37, 18900), dtype=np.float32
        )
        host_output1 = np.empty(
            (1, 32, 240, 240), dtype=np.float32
        )

        cuda.memcpy_htod_async(
            self.d_input, input_data, self.stream
        )

        success = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        if not success:
            raise RuntimeError("TensorRT inference failed")

        cuda.memcpy_dtoh_async(
            host_output0, self.d_output0, self.stream
        )
        cuda.memcpy_dtoh_async(
            host_output1, self.d_output1, self.stream
        )
        self.stream.synchronize()

        return self._decode_outputs(
            host_output0,
            host_output1,
            frame.shape,
        )

    # ==================================================
    # Decode Detection and Segmentation Outputs
    # ==================================================

    def _decode_outputs(self, output0, output1, frame_shape):
        frame_height, frame_width = frame_shape[:2]

        # output0 contains boxes, confidence scores and mask coefficients.
        pred = output0[0].transpose(1, 0)

        boxes_xywh = pred[:, 0:4]
        scores = pred[:, 4]
        mask_coeffs = pred[:, 5:37]

        keep = scores >= self.conf_threshold
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        mask_coeffs = mask_coeffs[keep]

        if len(scores) == 0:
            return []

        x_center = boxes_xywh[:, 0]
        y_center = boxes_xywh[:, 1]
        box_width = boxes_xywh[:, 2]
        box_height = boxes_xywh[:, 3]

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        boxes_for_nms = np.stack(
            [x1, y1, x2 - x1, y2 - y1],
            axis=1,
        )

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.asarray(indices).reshape(-1)
        if len(indices) > self.max_detections:
            best = np.argsort(scores[indices])[::-1][:self.max_detections]
            indices = indices[best]

        boxes = np.stack([x1, y1, x2, y2], axis=1)[indices]
        scores = scores[indices]
        mask_coeffs = mask_coeffs[indices]

        # output1 contains the 32 mask prototype channels.
        prototypes = output1[0].reshape(32, -1)
        mask_probabilities = self._sigmoid(
            mask_coeffs @ prototypes
        )
        mask_probabilities = mask_probabilities.reshape(
            -1, 240, 240
        )

        scale_x = frame_width / float(self.input_size)
        scale_y = frame_height / float(self.input_size)

        results = []

        for index, box in enumerate(boxes):
            mask = cv2.resize(
                mask_probabilities[index],
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = (
                mask >= self.mask_threshold
            ).astype(np.uint8) * 255

            bx1 = max(
                0, min(frame_width - 1, int(box[0] * scale_x))
            )
            by1 = max(
                0, min(frame_height - 1, int(box[1] * scale_y))
            )
            bx2 = max(
                0, min(frame_width, int(box[2] * scale_x))
            )
            by2 = max(
                0, min(frame_height, int(box[3] * scale_y))
            )

            cropped_mask = np.zeros_like(mask)

            if bx2 > bx1 and by2 > by1:
                cropped_mask[by1:by2, bx1:bx2] = (
                    mask[by1:by2, bx1:bx2]
                )

            results.append({
                "box": [bx1, by1, bx2, by2],
                "confidence": float(scores[index]),
                "mask": cropped_mask,
            })

        return results

    # ==================================================
    # Release GPU Resources
    # ==================================================

    def close(self):
        """Release GPU allocations and CUDA context."""
        if self.closed:
            return

        try:
            if hasattr(self, "stream"):
                self.stream.synchronize()

            for name in ("d_input", "d_output0", "d_output1"):
                buffer = getattr(self, name, None)
                if buffer is not None:
                    buffer.free()
                    setattr(self, name, None)

            self.context = None
            self.engine = None
            self.runtime = None

        finally:
            self.cuda_context.pop()
            self.cuda_context.detach()
            self.closed = True
