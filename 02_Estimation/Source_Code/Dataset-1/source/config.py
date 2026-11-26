# ======================================================
# Dataset Paths
# ======================================================

MODEL_DATASET_PATH = "output/model_dataset.xlsx"

FEATURE_DATASET_PATH = "output/feature_dataset.xlsx"

RAW_DATASET_PATH = "dataset/raw_dataset.xlsx"

# ======================================================
# YOLO Segmentation Paths
# ======================================================

YOLO_MODEL_PATH = "model/weights22k.pt"

IMAGE_FOLDER = "dataset/images"

MASK_OUTPUT_FOLDER = "output/masks"

REFINED_MASK_OUTPUT_FOLDER = "output/refined_masks"

TARGET_MASK_OUTPUT_FOLDER = "output/target_mask"

# ======================================================
# YOLO Segmentation
# ======================================================

YOLO_CONFIDENCE = 0.25

# ======================================================
# Mask Selection
# ======================================================

MIN_MASK_AREA = 3000
BORDER_MARGIN = 20

# ======================================================
# Mask Scoring
# ======================================================

AREA_SCORE_SCALE = 600
MAX_AREA_SCORE = 40
COMPLETENESS_THRESHOLD = 0.60

# ======================================================
# Feature Extraction Output
# ======================================================

OUTPUT_SKELETON_FOLDER = "output/raw_skeleton"

OUTPUT_MAIN_PATH_FOLDER = "output/main_path_skeleton"

# ======================================================
# Selected Features
# ======================================================

FEATURES = [
    "Area",
    "Perimeter",
    "Bounding_Box_Length",
    "Bounding_Box_Width",
    "Deviation",
    "Skeleton_Length",
    "Straight_Distance",
    "Straightness"
]

# ======================================================
# Dataset Split
# ======================================================

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.1765

# ======================================================
# Random Seed
# ======================================================

RANDOM_STATE = 42

# ======================================================
# ANN Parameters
# ======================================================

LEARNING_RATE = 0.001
EPOCHS = 3000
BATCH_SIZE = 4

HIDDEN_LAYER_1 = 16
HIDDEN_LAYER_2 = 8

# ======================================================
# Debug Mode
# ======================================================

DEBUG = True
