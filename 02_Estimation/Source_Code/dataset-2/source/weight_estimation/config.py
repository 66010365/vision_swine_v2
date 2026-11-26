# ============================================================
# Import Libraries
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# DATASET PATHS
# ============================================================

IMAGE_DATASET_PATH = (
    BASE_DIR / "dataset" / "PigRGB-Weight" / "RGB_9579"
)

FAILED_IMAGES_PATH = (
    BASE_DIR / "dataset" / "ภาพผิดหนัก"
)

WEIGHT_GROUND_TRUTH_PATH = (
    BASE_DIR / "dataset" / "Weight_Ground_Truth.xlsx"
)

USABLE_IMAGES_PATH = (
    BASE_DIR / "output" / "segmentation" / "segmentation_usable_dataset.xlsx"
)

MODEL_DATASET_PATH = (
    BASE_DIR / "output" / "feature_extraction" / "model_dataset.xlsx"
)

DATASET_CHECK_FOLDER = (
    BASE_DIR / "output" / "dataset_check"
)

RGB_MASK_MATCHING_PATH = (
    DATASET_CHECK_FOLDER / "rgb_mask_matching.xlsx"
)


# ============================================================
# YOLO SEGMENTATION
# ============================================================

YOLO_MODEL_PATH = BASE_DIR / "model" / "best.pt"

MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation" / "masks"
)

REFINED_MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation" / "refined_masks"
)

TARGET_MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation" / "target_mask"
)

SEGMENTATION_REVIEW_FOLDER = (
    BASE_DIR / "output" / "segmentation" / "segmentation_review"
)


# ============================================================
# SEGMENTATION TEST OUTPUT
# ============================================================

TEST_MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation_test" / "masks"
)

TEST_REFINED_MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation_test" / "refined_masks"
)

TEST_TARGET_MASK_OUTPUT_FOLDER = (
    BASE_DIR / "output" / "segmentation_test" / "target_masks"
)


# ============================================================
# YOLO PARAMETERS
# ============================================================

YOLO_CONFIDENCE = 0.90

YOLO_IMAGE_SIZE = 960


# ============================================================
# MASK SELECTION
# ============================================================

MIN_MASK_AREA = 3000

BORDER_MARGIN = 20


# ============================================================
# MASK SCORING
# ============================================================

AREA_SCORE_SCALE = 600

MAX_AREA_SCORE = 40

COMPLETENESS_THRESHOLD = 0.60


# ============================================================
# FEATURE EXTRACTION
# ============================================================

OUTPUT_SKELETON_FOLDER = (
    BASE_DIR / "output" / "feature_extraction" / "raw_skeleton"
)

OUTPUT_MAIN_PATH_FOLDER = (
    BASE_DIR / "output" / "feature_extraction" / "main_path_skeleton"
)

FEATURE_DATASET_PATH = (
    BASE_DIR / "output" / "feature_extraction" / "feature_dataset.xlsx"
)


# ============================================================
# SELECTED FEATURES
# ============================================================

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


# ============================================================
# DATASET SPLIT
# ============================================================

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.1765

KFOLD_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# ANN PARAMETERS
# ============================================================

LEARNING_RATE = 0.001

EPOCHS = 3000

BATCH_SIZE = 4

HIDDEN_LAYER_1 = 16

HIDDEN_LAYER_2 = 8

# ============================================================
# ESTIMATION MODEL OUTPUTS
# ============================================================

ESTIMATION_MODEL_PATH = (
    BASE_DIR / "output" / "estimation_model"
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

LINEAR_REGRESSION_SPLIT_PATH = (
    ESTIMATION_MODEL_PATH / "linear_regression_split"
)

LINEAR_REGRESSION_LOGO_PATH = (
    ESTIMATION_MODEL_PATH / "linear_regression_logo"
)

LINEAR_REGRESSION_KFOLD_PATH = (
    ESTIMATION_MODEL_PATH / "linear_regression_kfold"
)


# ============================================================
# POLYNOMIAL REGRESSION
# ============================================================

POLYNOMIAL_REGRESSION_SPLIT_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_regression_split"
)

POLYNOMIAL_REGRESSION_LOGO_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_regression_logo"
)

POLYNOMIAL_REGRESSION_KFOLD_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_regression_kfold"
)


# ============================================================
# ANN
# ============================================================

ANN_SPLIT_PATH = (
    ESTIMATION_MODEL_PATH / "ann_split"
)

ANN_LOGO_PATH = (
    ESTIMATION_MODEL_PATH / "ann_logo"
)

ANN_KFOLD_PATH = (
    ESTIMATION_MODEL_PATH / "ann_kfold"
)


# ============================================================
# POLYNOMIAL ANN
# ============================================================

POLYNOMIAL_ANN_SPLIT_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_ann_split"
)

POLYNOMIAL_ANN_LOGO_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_ann_logo"
)

POLYNOMIAL_ANN_KFOLD_PATH = (
    ESTIMATION_MODEL_PATH / "polynomial_ann_kfold"
)


# ============================================================
# DEBUG
# ============================================================

DEBUG = True