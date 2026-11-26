# ======================================================
# Import Libraries
# ======================================================

from pathlib import Path


# ======================================================
# Project Directory Configuration
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
REGRESSION_DIR = MODEL_DIR / "regression"

INPUT_DIR = PROJECT_ROOT / "test_images" / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"


# ======================================================
# AI Detection Model Paths
# ======================================================

PT_MODEL_PATH = MODEL_DIR / "weights22k.pt"
ONNX_MODEL_PATH = MODEL_DIR / "weights22k.onnx"
ENGINE_MODEL_PATH = MODEL_DIR / "weights22k.engine"


# ======================================================
# Weight Estimation Model Paths
# ======================================================

REGRESSION_MODEL_PATH = (
    REGRESSION_DIR / "linear_regression_split.joblib"
)

REGRESSION_SCALER_PATH = (
    REGRESSION_DIR / "linear_regression_split_scaler.joblib"
)


# ======================================================
# Test Video Paths
# ======================================================

INPUT_VIDEO_PATH = INPUT_DIR / "test2.mp4"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "ai_worker_result.mp4"


# ======================================================
# AI Detection Settings
# ======================================================

INPUT_SIZE = 960
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
MASK_THRESHOLD = 0.50


# ======================================================
# Video Processing Settings
# ======================================================

PROCESS_EVERY = 15
MAX_SECONDS = 10
