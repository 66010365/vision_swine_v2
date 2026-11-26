# ======================================================
# Import Libraries
# ======================================================

import os


# ======================================================
# Dataset Configuration
# ======================================================

RGB_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "RGB_MASK_3394",
    "RGB_3394"
)

LABEL_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "transfer_learning_dataset",
    "raw_labels"
)


# ======================================================
# Get RGB Images
# ======================================================

image_files = []

for root, dirs, files in os.walk(
    RGB_DATASET_PATH
):

    for file in files:

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            image_files.append(
                os.path.join(
                    root,
                    file
                )
            )

image_files = sorted(
    image_files
)


# ======================================================
# Get YOLO Labels
# ======================================================

label_files = []

for root, dirs, files in os.walk(
    LABEL_DATASET_PATH
):

    for file in files:

        if file.lower().endswith(
            ".txt"
        ):

            label_files.append(
                os.path.join(
                    root,
                    file
                )
            )

label_files = sorted(
    label_files
)


# ======================================================
# Create File Names
# ======================================================

image_names = {
    os.path.splitext(
        os.path.basename(file)
    )[0]
    for file in image_files
}

label_names = {
    os.path.splitext(
        os.path.basename(file)
    )[0]
    for file in label_files
}


# ======================================================
# Compare Images and Labels
# ======================================================

missing_labels = (
    image_names - label_names
)

missing_images = (
    label_names - image_names
)


# ======================================================
# Check Empty Labels
# ======================================================

empty_labels = []

for label_file in label_files:

    if os.path.getsize(
        label_file
    ) == 0:

        empty_labels.append(
            os.path.splitext(
                os.path.basename(label_file)
            )[0]
        )


# ======================================================
# Check YOLO Label Format
# ======================================================

invalid_labels = []

for label_file in label_files:

    label_name = os.path.splitext(
        os.path.basename(label_file)
    )[0]

    try:

        with open(
            label_file,
            "r"
        ) as file:

            lines = file.readlines()

        for line in lines:

            values = line.strip().split()

            # Class ID + at least 3 polygon points
            if len(values) < 7:

                invalid_labels.append(
                    label_name
                )

                break

            class_id = int(
                values[0]
            )

            # Check Class ID
            if class_id != 0:

                invalid_labels.append(
                    label_name
                )

                break

            coordinates = list(
                map(
                    float,
                    values[1:]
                )
            )

            # Coordinates must be pairs
            if len(coordinates) % 2 != 0:

                invalid_labels.append(
                    label_name
                )

                break

            # Normalized coordinates
            for value in coordinates:

                if value < 0 or value > 1:

                    invalid_labels.append(
                        label_name
                    )

                    break

            else:

                continue

            break

    except Exception:

        invalid_labels.append(
            label_name
        )


# ======================================================
# Display Result
# ======================================================

print("=" * 60)
print("SEGMENTATION LABEL DATASET CHECK")
print("=" * 60)

print()

print("Dataset")
print("-" * 60)

print(
    f"RGB images   : "
    f"{len(image_files)}"
)

print(
    f"YOLO labels  : "
    f"{len(label_files)}"
)

print()

print("Matching")
print("-" * 60)

print(
    f"Matched pairs : "
    f"{len(image_names & label_names)}"
)

print(
    f"Missing label : "
    f"{len(missing_labels)}"
)

print(
    f"Missing RGB   : "
    f"{len(missing_images)}"
)

print()

print("Label Check")
print("-" * 60)

print(
    f"Empty labels  : "
    f"{len(empty_labels)}"
)

print(
    f"Invalid labels: "
    f"{len(invalid_labels)}"
)


# ======================================================
# Display Missing Labels
# ======================================================

if missing_labels:

    print()
    print("RGB without Label")
    print("-" * 60)

    for name in sorted(
        missing_labels
    ):

        print(
            name
        )


# ======================================================
# Display Missing Images
# ======================================================

if missing_images:

    print()
    print("Label without RGB")
    print("-" * 60)

    for name in sorted(
        missing_images
    ):

        print(
            name
        )


# ======================================================
# Display Empty Labels
# ======================================================

if empty_labels:

    print()
    print("Empty Labels")
    print("-" * 60)

    for name in sorted(
        empty_labels
    ):

        print(
            name
        )


# ======================================================
# Display Invalid Labels
# ======================================================

if invalid_labels:

    print()
    print("Invalid Labels")
    print("-" * 60)

    for name in sorted(
        set(invalid_labels)
    ):

        print(
            name
        )


# ======================================================
# Final Result
# ======================================================

print()
print("=" * 60)

if (
    len(missing_labels) == 0
    and
    len(missing_images) == 0
    and
    len(empty_labels) == 0
    and
    len(invalid_labels) == 0
):

    print(
        "SEGMENTATION LABEL CHECK PASSED"
    )

else:

    print(
        "SEGMENTATION LABEL CHECK FOUND PROBLEMS"
    )

print("=" * 60)