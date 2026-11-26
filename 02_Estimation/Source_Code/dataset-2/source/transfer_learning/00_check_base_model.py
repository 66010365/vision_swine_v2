# ======================================================
# Import Libraries
# ======================================================

from ultralytics import YOLO


# ======================================================
# Model Configuration
# ======================================================

YOLO_MODEL_PATH = (
    "model/weights22k.pt"
)


# ======================================================
# Load Model
# ======================================================

model = YOLO(
    YOLO_MODEL_PATH
)


# ======================================================
# Display Model Information
# ======================================================

print("=" * 60)
print("YOLO MODEL CHECK")
print("=" * 60)

print()

print(
    f"Model path : {YOLO_MODEL_PATH}"
)

print(
    f"Task       : {model.task}"
)

print(
    f"Model type : "
    f"{type(model.model).__name__}"
)

print()

print(
    "Class names:"
)

print(
    model.names
)

print()

print(
    f"Number of classes : "
    f"{len(model.names)}"
)

print()

print("=" * 60)
print("YOLO MODEL CHECK COMPLETED")
print("=" * 60)