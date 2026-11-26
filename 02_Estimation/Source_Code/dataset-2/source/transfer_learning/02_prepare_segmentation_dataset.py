# ======================================================
# Import Libraries
# ======================================================

import os
import cv2
import numpy as np


# ======================================================
# Dataset Configuration
# ======================================================

MASK_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "RGB_MASK_3394",
    "MASK_3394"
)

# ======================================================
# Output Folders
# ======================================================

OUTPUT_FOLDER = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "transfer_learning_dataset"
)

LABEL_OUTPUT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "raw_labels"
)

VISUALIZATION_OUTPUT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "segmentation_visualization"
)

MIN_CONTOUR_AREA = 100


# ======================================================
# Create Output Folder
# ======================================================

os.makedirs(
    LABEL_OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    VISUALIZATION_OUTPUT_FOLDER,
    exist_ok=True
)


# ======================================================
# Get Mask Files
# ======================================================

mask_files = []

for root, dirs, files in os.walk(
    MASK_DATASET_PATH
):

    for file in files:

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            mask_files.append(
                os.path.join(
                    root,
                    file
                )
            )

mask_files = sorted(
    mask_files
)


# ======================================================
# Display Dataset Information
# ======================================================

print("=" * 60)
print("SEGMENTATION LABEL PREPARATION")
print("=" * 60)

print(
    f"Total mask images : {len(mask_files)}"
)


# ======================================================
# Process All Images
# ======================================================

for mask_file in mask_files:

    # ==================================================
    # Image Name
    # ==================================================

    image_name = os.path.splitext(
        os.path.basename(mask_file)
    )[0]

    # ==================================================
    # Read Mask
    # ==================================================

    mask = cv2.imread(
        mask_file,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        print(
            f"Invalid mask : {mask_file}"
        )

        continue

    # ==================================================
    # Threshold Mask
    # ==================================================

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # ==================================================
    # Find Contours
    # ==================================================

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ==================================================
    # Image Information
    # ==================================================

    height, width = binary_mask.shape

    polygons = []

    # ==================================================
    # Convert Contours to YOLO Polygon
    # ==================================================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_CONTOUR_AREA:

            continue

        # ==============================================
        # Simplify Polygon
        # ==============================================

        epsilon = 0.002 * cv2.arcLength(
            contour,
            True
        )

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        # ==============================================
        # Normalize Coordinates
        # ==============================================

        points = []

        for point in polygon:

            x, y = point[0]

            x_normalized = x / width
            y_normalized = y / height

            points.extend(
                [
                    x_normalized,
                    y_normalized
                ]
            )

        # ==============================================
        # Check Polygon
        # ==============================================

        if len(points) >= 6:

            polygons.append(
                points
            )

    # ==================================================
    # Create YOLO Label
    # ==================================================

    image_name = os.path.splitext(
        os.path.basename(mask_file)
    )[0]

    label_path = os.path.join(
        LABEL_OUTPUT_FOLDER,
        image_name + ".txt"
    )

    with open(
        label_path,
        "w"
    ) as label_file:

        for polygon in polygons:

            label_file.write(
                "0 "
                + " ".join(
                    f"{value:.6f}"
                    for value in polygon
                )
                + "\n"
            )

    # ==================================================
    # Create Visualization
    # ==================================================

    visualization = cv2.cvtColor(
        binary_mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.drawContours(
        visualization,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    output_image_path = os.path.join(
        VISUALIZATION_OUTPUT_FOLDER,
        image_name + "_check.jpg"
    )

    cv2.imwrite(
        output_image_path,
        visualization
    )

    # ==================================================
    # Display Result
    # ==================================================

    print(
        f"{mask_file} -> "
        f"{len(polygons)} polygon(s)"
    )


# ======================================================
# Completion
# ======================================================

print()
print("=" * 60)
print("SEGMENTATION LABEL PREPARATION COMPLETED")
print("=" * 60)

print(
    f"Output folder : {OUTPUT_FOLDER}"
)