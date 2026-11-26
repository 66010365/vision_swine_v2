# ======================================================
# Import Libraries
# ======================================================

from ultralytics import YOLO


# ======================================================
# Model Configuration
# ======================================================

MODEL_PATH = (
    "model/weights22k.pt"
)


# ======================================================
# Dataset Configuration
# ======================================================

DATA_YAML_PATH = (
    "dataset/PigRGB-Weight/"
    "transfer_learning_dataset/data.yaml"
)


# ======================================================
# Training Configuration
# ======================================================

EPOCHS = 20

IMAGE_SIZE = 960

BATCH_SIZE = 2

DEVICE = 0

WORKERS = 0

PROJECT_NAME = (
    "output/transfer_learning"
)

RUN_NAME = (
    "pig_segmentation_test"
)


# ======================================================
# Main
# ======================================================

def main():

    # ==================================================
    # Display Training Information
    # ==================================================

    print("=" * 60)
    print("YOLO SEGMENTATION TRANSFER LEARNING")
    print("=" * 60)

    print()

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Dataset : {DATA_YAML_PATH}"
    )

    print(
        f"Epochs  : {EPOCHS}"
    )

    print(
        f"Image   : {IMAGE_SIZE}"
    )

    print(
        f"Batch   : {BATCH_SIZE}"
    )

    print()


    # ==================================================
    # Load Pretrained Model
    # ==================================================

    model = YOLO(
        MODEL_PATH
    )


    # ==================================================
    # Train Model
    # ==================================================

    results = model.train(

        data=DATA_YAML_PATH,

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        rect=True,

        project=PROJECT_NAME,

        name=RUN_NAME,

        pretrained=True,

        exist_ok=True

    )


    # ==================================================
    # Training Completed
    # ==================================================

    print()

    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Results saved to: "
        f"{PROJECT_NAME}/{RUN_NAME}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":

    main()