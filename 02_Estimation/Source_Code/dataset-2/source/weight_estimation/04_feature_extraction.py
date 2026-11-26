# ======================================================
# Import Libraries
# ======================================================

import os
import re
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


# ======================================================
# Import Configuration
# ======================================================

from config import (
    TARGET_MASK_OUTPUT_FOLDER,
    USABLE_IMAGES_PATH,
    OUTPUT_SKELETON_FOLDER,
    OUTPUT_MAIN_PATH_FOLDER,
    FEATURE_DATASET_PATH,
    DEBUG
)


# ======================================================
# Read Mask
# ======================================================

def read_mask(mask_path):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    return mask


# ======================================================
# Get Largest Contour
# ======================================================

def get_largest_contour(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    return largest_contour


# ======================================================
# Size Features
# ======================================================

# Calculate Area
def calculate_area(mask):

    area = cv2.countNonZero(mask)

    return area


# Calculate Perimeter
def calculate_perimeter(contour):

    if contour is None:
        return 0

    perimeter = cv2.arcLength(
        contour,
        True
    )

    return perimeter


# Calculate Bounding Box Length
def calculate_bounding_box_length(contour):

    if contour is None:
        return 0

    bounding_rect = cv2.minAreaRect(
        contour
    )

    bounding_width, bounding_height = bounding_rect[1]

    bounding_box_length = max(
        bounding_width,
        bounding_height
    )

    return bounding_box_length


# Calculate Bounding Box Width
def calculate_bounding_box_width(contour):

    if contour is None:
        return 0

    bounding_rect = cv2.minAreaRect(
        contour
    )

    bounding_width, bounding_height = bounding_rect[1]

    bounding_box_width = min(
        bounding_width,
        bounding_height
    )

    return bounding_box_width


# ======================================================
# Position Features
# ======================================================

# Calculate Deviation
def calculate_deviation(mask, contour):

    if contour is None:
        return 0

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return 0

    # Pig centroid
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]

    # Image center
    image_height, image_width = mask.shape

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    # Distance
    distance = np.sqrt(
        (centroid_x - image_center_x) ** 2 +
        (centroid_y - image_center_y) ** 2
    )

    # Maximum distance
    max_distance = np.sqrt(
        image_center_x ** 2 +
        image_center_y ** 2
    )

    deviation = distance / max_distance

    return deviation


# ======================================================
# Skeleton Features
# ======================================================

# Create Skeleton
def create_skeleton(mask):

    # Convert to binary
    binary_mask = mask > 0

    # Skeletonization
    skeleton = skeletonize(
        binary_mask
    )

    # Convert to uint8
    skeleton = (
        skeleton.astype(np.uint8)
        * 255
    )

    return skeleton


# Analyze Skeleton
def analyze_skeleton(skeleton):

    skeleton_graph = Skeleton(
        skeleton > 0
    )

    branch_data = summarize(
        skeleton_graph,
        separator="-"
    )

    return skeleton_graph


# Find Main Path
def find_main_path(
    skeleton,
    image_name
):

    skeleton_graph = Skeleton(
        skeleton > 0
    )

    branch_data = summarize(
        skeleton_graph,
        separator="-"
    )

    # Find longest path
    longest_path_index = branch_data[
        "branch-distance"
    ].idxmax()

    # Longest path information
    longest_path = branch_data.loc[
        longest_path_index
    ]

    # Straight distance
    straight_distance = longest_path[
        "euclidean-distance"
    ]

    # Path length
    path_length = longest_path[
        "branch-distance"
    ]

    # Get main path coordinates
    main_path_coordinates = (
        skeleton_graph.path_coordinates(
            longest_path_index
        )
    )

    # Create main path image
    main_path_image = np.zeros_like(
        skeleton
    )

    # Draw main path
    for y, x in main_path_coordinates:

        main_path_image[
            int(y),
            int(x)
        ] = 255

    print_main_path_summary(
        image_name,
        branch_data,
        longest_path_index,
        len(main_path_coordinates)
    )

    return (
        main_path_image,
        path_length,
        straight_distance
    )


# Calculate Straightness
def calculate_straightness(
    skeleton_length,
    straight_distance
):

    if skeleton_length == 0:
        return 0

    straightness = (
        straight_distance
        / skeleton_length
    )

    return straightness


# ======================================================
# Debug Functions
# ======================================================

def print_main_path_summary(
    image_name,
    branch_data,
    longest_path_index,
    main_path_pixels
):

    if not DEBUG:
        return

    longest_path = branch_data.loc[
        longest_path_index
    ]

    print("=" * 50)

    print(f"Image : {image_name}")

    print("\nBranch Summary")
    print("-" * 34)

    print(
        f"Total Branches : {len(branch_data)}"
    )

    print("\nMain Path")
    print("-" * 34)

    print(
        f"Path Index          : {longest_path_index}"
    )

    print(
        f"Skeleton Length     : "
        f"{longest_path['branch-distance']:.2f} px"
    )

    print(
        f"Straight Distance   : "
        f"{longest_path['euclidean-distance']:.2f} px"
    )

    print(
        "Main Path Pixels    : "
        f"{main_path_pixels}"
    )

    print("=" * 50)


# ======================================================
# Find Target Mask Files
# ======================================================

def normalize_mask_name(filename):

    """
    Convert Target Mask filename to original image name.

    Example:

    image001_target_1.png
        ->
    image001
    """

    stem = Path(filename).stem

    stem = re.sub(
        r"_target_\d+$",
        "",
        stem
    )

    return stem


def get_target_mask_files():

    """
    Search all Target Mask files recursively
    and create a dictionary using image name.
    """

    target_mask_files = {}

    for root, dirs, files in os.walk(
        TARGET_MASK_OUTPUT_FOLDER
    ):

        for file in files:

            if not file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            image_name = normalize_mask_name(
                file
            )

            full_path = os.path.join(
                root,
                file
            )

            if image_name not in target_mask_files:

                target_mask_files[
                    image_name
                ] = []

            target_mask_files[
                image_name
            ].append(full_path)

    return target_mask_files


# ======================================================
# Extract Features
# ======================================================

def extract_features(
    mask,
    skeleton,
    image_name
):

    # ======================================================
    # Size Features
    # ======================================================

    area = calculate_area(mask)

    contour = get_largest_contour(mask)

    perimeter = calculate_perimeter(
        contour
    )

    bounding_box_length = (
        calculate_bounding_box_length(
            contour
        )
    )

    bounding_box_width = (
        calculate_bounding_box_width(
            contour
        )
    )

    # ======================================================
    # Position Features
    # ======================================================

    deviation = calculate_deviation(
        mask,
        contour
    )

    # ======================================================
    # Skeleton Features
    # ======================================================

    skeleton_graph = analyze_skeleton(
        skeleton
    )

    main_path_image, path_length, straight_distance = (
        find_main_path(
            skeleton,
            image_name
        )
    )

    skeleton_length = path_length

    straightness = calculate_straightness(
        path_length,
        straight_distance
    )

    features = {

        "Area": area,

        "Perimeter": perimeter,

        "Bounding_Box_Length":
            bounding_box_length,

        "Bounding_Box_Width":
            bounding_box_width,

        "Deviation":
            deviation,

        "Skeleton_Length":
            skeleton_length,

        "Straight_Distance":
            straight_distance,

        "Straightness":
            straightness
    }

    return (
        features,
        main_path_image
    )


# ======================================================
# Save Features Excel
# ======================================================

def save_features_excel(
    all_features,
    output_excel
):

    if len(all_features) == 0:
        return

    df = pd.DataFrame(
        all_features
    )

    df.to_excel(
        output_excel,
        index=False
    )


# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Read Usable Dataset
    # ======================================================

    df = pd.read_excel(
        USABLE_IMAGES_PATH
    )

    print("=" * 60)
    print("การสกัดคุณลักษณะ")
    print("=" * 60)

    print(
        f"จำนวนภาพที่ต้องประมวลผล : {len(df):,}"
    )


    # ======================================================
    # Find Target Mask Files
    # ======================================================

    print()
    print("กำลังค้นหา Target Mask...")

    target_mask_files = (
        get_target_mask_files()
    )

    print(
        f"พบ Target Mask           : {sum(len(v) for v in target_mask_files.values()):,}"
    )

    print(
        f"ชื่อภาพที่ไม่ซ้ำ           : {len(target_mask_files):,}"
    )


    # ======================================================
    # Create Output Folders
    # ======================================================

    os.makedirs(
        OUTPUT_SKELETON_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_MAIN_PATH_FOLDER,
        exist_ok=True
    )


    # ======================================================
    # Extract Features
    # ======================================================

    all_features = []

    processed_count = 0
    skipped_count = 0

    print()
    print("กำลังสกัดคุณลักษณะ...")

    for index, row in df.iterrows():

        image_name = row[
            "Image Name"
        ]


        # --------------------------------------------------
        # Find Target Mask
        # --------------------------------------------------

        matches = target_mask_files.get(
            image_name,
            []
        )


        # --------------------------------------------------
        # Target Mask Not Found
        # --------------------------------------------------

        if len(matches) == 0:

            print()
            print(
                "[WARNING] Target Mask not found:"
            )

            print(
                image_name
            )

            skipped_count += 1

            continue


        # --------------------------------------------------
        # Multiple Target Masks Found
        # --------------------------------------------------

        if len(matches) > 1:

            print()
            print(
                "[WARNING] Multiple Target Masks found:"
            )

            print(
                image_name
            )

            for path in matches:
                print(
                    f"  - {path}"
                )

            skipped_count += 1

            continue


        # --------------------------------------------------
        # Select Target Mask
        # --------------------------------------------------

        mask_path = matches[0]




        # --------------------------------------------------
        # Check Mask
        # --------------------------------------------------

        if not os.path.exists(
            mask_path
        ):

            print(
                "[WARNING] Mask not found:"
            )

            print(
                mask_path
            )

            skipped_count += 1

            continue


        # --------------------------------------------------
        # Read Mask
        # --------------------------------------------------

        mask = read_mask(
            mask_path
        )


        if mask is None:

            print(
                "[WARNING] Cannot read mask:"
            )

            print(
                mask_path
            )

            skipped_count += 1

            continue


        # --------------------------------------------------
        # Create Skeleton
        # --------------------------------------------------

        skeleton = create_skeleton(
            mask
        )


        # ==================================================
        # Determine Fold and Pig Folder
        # ==================================================

        mask_folder = os.path.dirname(
            mask_path
        )

        pig_folder = os.path.basename(
            mask_folder
        )

        fold_folder = os.path.basename(
            os.path.dirname(
                mask_folder
            )
        )


        # --------------------------------------------------
        # Create Skeleton Output Folder
        # --------------------------------------------------

        skeleton_output_folder = os.path.join(
            OUTPUT_SKELETON_FOLDER,
            fold_folder,
            pig_folder
        )

        os.makedirs(
            skeleton_output_folder,
            exist_ok=True
        )


        # --------------------------------------------------
        # Save Skeleton
        # --------------------------------------------------

        skeleton_path = os.path.join(
            skeleton_output_folder,
            os.path.basename(mask_path)
        )

        cv2.imwrite(
            skeleton_path,
            skeleton
        )


        # --------------------------------------------------
        # Extract Features
        # --------------------------------------------------

        features, main_path_image = (
            extract_features(
                mask,
                skeleton,
                image_name
            )
        )


        # --------------------------------------------------
        # Add Image Information
        # --------------------------------------------------

        features = {

            "Image":
                image_name,

            "Image_Path":
                row["Image Path"],

            "Mask_Path":
                mask_path,

            **features
        }


        # ==================================================
        # Create Main Path Output Folder
        # ==================================================

        main_path_output_folder = os.path.join(
            OUTPUT_MAIN_PATH_FOLDER,
            fold_folder,
            pig_folder
        )

        os.makedirs(
            main_path_output_folder,
            exist_ok=True
        )


        # --------------------------------------------------
        # Save Main Path Skeleton
        # --------------------------------------------------

        main_path_skeleton_path = os.path.join(
            main_path_output_folder,
            os.path.basename(mask_path)
        )

        cv2.imwrite(
            main_path_skeleton_path,
            main_path_image
        )


        # --------------------------------------------------
        # Add Features
        # --------------------------------------------------

        all_features.append(
            features
        )

        processed_count += 1


    # ======================================================
    # Save Feature Dataset
    # ======================================================

    save_features_excel(
        all_features,
        FEATURE_DATASET_PATH
    )


    # ======================================================
    # Final Result
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผลการสกัดคุณลักษณะ")
    print("=" * 60)

    print(
        f"จำนวนภาพทั้งหมด          : {len(df):,} ภาพ"
    )

    print(
        f"ประมวลผลสำเร็จ            : {processed_count:,} ภาพ"
    )

    print(
        f"ข้าม                      : {skipped_count:,} ภาพ"
    )

    print()
    print("Feature ที่ได้             : 8 ตัว")

    print(
        f"บันทึกไฟล์ไว้ที่            : {FEATURE_DATASET_PATH}"
    )

    print("=" * 60)
    print("Feature Extraction completed.")
    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()