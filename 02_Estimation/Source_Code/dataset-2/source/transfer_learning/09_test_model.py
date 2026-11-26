# ======================================================
# Import Libraries
# ======================================================

import os

from ultralytics import YOLO


# ======================================================
# Model Configuration
# ======================================================

MODEL_PATH = os.path.join(
    "runs",
    "segment",
    "output",
    "transfer_learning",
    "pig_segmentation_test",
    "weights",
    "best.pt"
)


# ======================================================
# Dataset Configuration
# ======================================================

DATA_YAML_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "transfer_learning_dataset",
    "data.yaml"
)


# ======================================================
# Test Configuration
# ======================================================

IMAGE_SIZE = 960

BATCH_SIZE = 2

DEVICE = 0

WORKERS = 0

PROJECT_NAME = os.path.join(
    "output",
    "test"
)

RUN_NAME = "pig_segmentation_test"


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 60)
    print("YOLO SEGMENTATION MODEL TEST")
    print("=" * 60)

    print()

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Dataset : {DATA_YAML_PATH}"
    )

    print(
        f"Image   : {IMAGE_SIZE}"
    )

    print(
        f"Batch   : {BATCH_SIZE}"
    )

    print()

    # ==================================================
    # Check Model
    # ==================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "ERROR: Model not found."
        )

        return

    # ==================================================
    # Check Dataset
    # ==================================================

    if not os.path.exists(
        DATA_YAML_PATH
    ):

        print(
            "ERROR: Dataset YAML not found."
        )

        return

    # ==================================================
    # Load Model
    # ==================================================

    model = YOLO(
        MODEL_PATH
    )

    # ==================================================
    # Test Model
    # ==================================================

    results = model.val(

        data=DATA_YAML_PATH,

        split="test",

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        project=PROJECT_NAME,

        name=RUN_NAME,

        plots=True

    )

    # ==================================================
    # Display Results
    # ==================================================

    print()

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

    print()

    print(
        f"Box Precision : "
        f"{results.box.mp:.4f}"
    )

    print(
        f"Box Recall    : "
        f"{results.box.mr:.4f}"
    )

    print(
        f"Box mAP50     : "
        f"{results.box.map50:.4f}"
    )

    print(
        f"Box mAP50-95  : "
        f"{results.box.map:.4f}"
    )

    print()

    print(
        f"Mask Precision : "
        f"{results.seg.mp:.4f}"
    )

    print(
        f"Mask Recall    : "
        f"{results.seg.mr:.4f}"
    )

    print(
        f"Mask mAP50     : "
        f"{results.seg.map50:.4f}"
    )

    print(
        f"Mask mAP50-95  : "
        f"{results.seg.map:.4f}"
    )

    print()

    print(
        f"Results saved to : "
        f"{PROJECT_NAME}/{RUN_NAME}"
    )

    print()

    print("=" * 60)
    print("YOLO SEGMENTATION TEST COMPLETED")
    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":

    main()