# ======================================================
# Import Libraries
# ======================================================

import os

from ultralytics import YOLO


# ======================================================
# Model Configuration
# ======================================================

MODEL_PATH = os.path.join(
    "output",
    "transfer_learning",
    "pig_segmentation_test",
    "weights",
    "best.pt"
)

# ======================================================
# Display Header
# ======================================================

print("=" * 60)
print("TRAINED MODEL CHECK")
print("=" * 60)

print()


# ======================================================
# Check Model File
# ======================================================

print(
    f"Model path : {MODEL_PATH}"
)

print()


if not os.path.exists(
    MODEL_PATH
):

    print(
        "ERROR: Trained model not found."
    )

    print()

    print(
        f"Expected path : {MODEL_PATH}"
    )

    exit()


# ======================================================
# Load Trained Model
# ======================================================

model = YOLO(
    MODEL_PATH
)


# ======================================================
# Display Model Information
# ======================================================

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


# ======================================================
# Check Model Task
# ======================================================

if model.task == "segment":

    print()

    print(
        "Task check  : PASSED"
    )

else:

    print()

    print(
        "Task check  : FAILED"
    )


# ======================================================
# Check Class
# ======================================================

if (
    len(model.names) == 1
    and model.names.get(0) == "pig"
):

    print(
        "Class check : PASSED"
    )

else:

    print(
        "Class check : FAILED"
    )


# ======================================================
# Completion
# ======================================================

print()

print("=" * 60)
print("TRAINED MODEL CHECK COMPLETED")
print("=" * 60)