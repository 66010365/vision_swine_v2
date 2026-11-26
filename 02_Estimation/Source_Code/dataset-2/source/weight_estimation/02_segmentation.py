# ======================================================
# Import Libraries
# ======================================================

import os

import cv2
import numpy as np
from ultralytics import YOLO


# ======================================================
# Import Configuration
# ======================================================

from config import (
    YOLO_MODEL_PATH,
    IMAGE_DATASET_PATH,
    MASK_OUTPUT_FOLDER,
    REFINED_MASK_OUTPUT_FOLDER,
    TARGET_MASK_OUTPUT_FOLDER,
    SEGMENTATION_REVIEW_FOLDER,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    MIN_MASK_AREA,
    BORDER_MARGIN,
    AREA_SCORE_SCALE,
    MAX_AREA_SCORE,
    COMPLETENESS_THRESHOLD,
    DEBUG
)


# ======================================================
# Test Configuration
# ======================================================

TEST_MODE = False

TEST_IMAGE_COUNT = 1


# ======================================================
# Load YOLO Segmentation Model
# ======================================================

def load_model(model_path):

    return YOLO(model_path)


# ======================================================
# Read Image
# ======================================================

def read_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    return image


# ======================================================
# Segment Image
# ======================================================

def segment_image(model, image):

    results = model(
        image,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        retina_masks=True,
        device=0
    )

    return results


# ======================================================
# Get Pig Masks
# ======================================================

def get_pig_masks(results):

    if results[0].masks is None:
        return []

    masks = results[0].masks.data.cpu().numpy()

    return masks


# ======================================================
# Refine Mask
# ======================================================

def refine_mask(mask):

    # Convert binary mask (0/1) to OpenCV mask (0/255)

    if mask.max() <= 1:

        mask = (
            mask * 255
        ).astype(np.uint8)

    else:

        mask = mask.astype(np.uint8)


    # Find external contours

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:

        return mask


    # Debug

    if DEBUG:

        print(
            f"Number of contours : "
            f"{len(contours)}"
        )

        for i, contour in enumerate(contours):

            print(
                f"Contour {i}: "
                f"Area = "
                f"{cv2.contourArea(contour)}"
            )


    # Find largest contour

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    # Create empty mask

    refined_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )


    # Draw only largest contour

    cv2.drawContours(
        refined_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED
    )


    return refined_mask


# ======================================================
# Refine Multiple Masks
# ======================================================

def refine_masks(masks):

    refined_masks = []

    for mask in masks:

        refined = refine_mask(mask)

        refined_masks.append(refined)

    return refined_masks


# ======================================================
# Calculate Mask Score
# ======================================================

def calculate_mask_score(mask):

    # ----------------------------
    # Area
    # ----------------------------

    area = cv2.countNonZero(mask)


    # ----------------------------
    # Find contour
    # ----------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    M = cv2.moments(
        largest_contour
    )


    if M["m00"] == 0:

        return 0


    cx = int(
        M["m10"] / M["m00"]
    )

    cy = int(
        M["m01"] / M["m00"]
    )


    # ----------------------------
    # Image center
    # ----------------------------

    height, width = mask.shape

    image_center_x = width // 2
    image_center_y = height // 2


    # ----------------------------
    # Distance to image center
    # ----------------------------

    distance = np.sqrt(
        (cx - image_center_x) ** 2 +
        (cy - image_center_y) ** 2
    )


    max_distance = np.sqrt(
        image_center_x ** 2 +
        image_center_y ** 2
    )


    # ----------------------------
    # Border Score (0-20)
    # ----------------------------

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )


    if (
        x <= BORDER_MARGIN
        or y <= BORDER_MARGIN
        or (x + w) >= (
            width - BORDER_MARGIN
        )
        or (y + h) >= (
            height - BORDER_MARGIN
        )
    ):

        border_score = 0

    else:

        border_score = 20


    # ----------------------------
    # Completeness Score (0-10)
    # ----------------------------

    bounding_area = w * h


    if bounding_area == 0:

        completeness_score = 0

    else:

        fill_ratio = area / bounding_area


        if (
            fill_ratio >=
            COMPLETENESS_THRESHOLD
        ):

            completeness_score = 10

        else:

            completeness_score = 0


    # ----------------------------
    # Area Score (0-40)
    # ----------------------------

    area_score = min(
        area / AREA_SCORE_SCALE,
        MAX_AREA_SCORE
    )


    # ----------------------------
    # Center Score (0-30)
    # ----------------------------

    center_score = (
        1 - distance / max_distance
    ) * 30


    # ----------------------------
    # Total Score
    # ----------------------------

    score = (
        area_score
        + center_score
        + border_score
        + completeness_score
    )


    if DEBUG:

        print(
            f"Area         : {area}"
        )

        print(
            f"Center       : "
            f"({cx}, {cy})"
        )

        print(
            f"Area Score   : "
            f"{area_score:.2f}"
        )

        print(
            f"Center Score : "
            f"{center_score:.2f}"
        )

        print(
            f"Border Score : "
            f"{border_score:.2f}"
        )

        print(
            f"Completeness : "
            f"{completeness_score:.2f}"
        )

        print(
            f"Total Score  : "
            f"{score:.2f}"
        )


    return score


# ======================================================
# Select Target Mask
# ======================================================

def select_target_mask(refined_masks):

    best_score = -1

    best_mask = None


    for i, mask in enumerate(refined_masks):

        area = cv2.countNonZero(mask)


        if area < MIN_MASK_AREA:

            continue


        score = calculate_mask_score(mask)


        if DEBUG:

            print(
                f"Mask {i+1} Score : "
                f"{score:.2f}"
            )


        if score > best_score:

            best_score = score
            best_mask = mask


    if DEBUG:

        print(
            f"Selected Score : "
            f"{best_score:.2f}"
        )


    return best_mask


# ======================================================
# Save Masks
# ======================================================

def save_masks(
    masks,
    output_folder,
    image_name,
    prefix="mask"
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    for i, mask in enumerate(masks):

        # Convert binary mask (0/1) -> (0/255)

        if mask.max() <= 1:

            mask_to_save = (
                mask * 255
            ).astype(np.uint8)

        else:

            mask_to_save = (
                mask.astype(np.uint8)
            )


        file_name = os.path.join(
            output_folder,
            f"{image_name}_{prefix}_{i+1}.png"
        )


        cv2.imwrite(
            file_name,
            mask_to_save
        )


# ======================================================
# Segment Pigs
# ======================================================

def segment_pigs(model, image):

    # Step 1 : Segment image using YOLO

    results = segment_image(
        model,
        image
    )


    # Step 2 : Get all pig masks

    masks = get_pig_masks(
        results
    )


    # Step 3 : Refine all masks

    refined_masks = refine_masks(
        masks
    )


    return (
        masks,
        refined_masks
    )


# ======================================================
# Create Segmentation Review Image
# ======================================================

def create_review_image(image, mask):

    # Convert mask to binary

    if mask.max() <= 1:

        binary_mask = (
            mask * 255
        ).astype(np.uint8)

    else:

        binary_mask = (
            mask.astype(np.uint8)
        )


    # Create colored overlay

    overlay = image.copy()


    overlay[
        binary_mask > 0
    ] = (
        0.5 * overlay[
            binary_mask > 0
        ]
        +
        0.5 * np.array(
            [255, 0, 0]
        )
    ).astype(np.uint8)


    # Combine original image and overlay

    review = cv2.addWeighted(
        image,
        0.5,
        overlay,
        0.5,
        0
    )


    return review


# ======================================================
# Main
# ======================================================

def main():

    # ==================================================
    # Load YOLO Model
    # ==================================================

    model = load_model(
        YOLO_MODEL_PATH
    )


    # ==================================================
    # Initialize Counters
    # ==================================================

    total_images = 0
    processed_images = 0
    no_mask_images = 0
    no_refined_mask_images = 0
    no_target_mask_images = 0


    # ==================================================
    # Get Fold Folders
    # ==================================================

    fold_folders = sorted(
        [
            folder
            for folder in os.listdir(
                IMAGE_DATASET_PATH
            )
            if os.path.isdir(
                os.path.join(
                    IMAGE_DATASET_PATH,
                    folder
                )
            )
        ]
    )


    if DEBUG:

        print(
            f"Found "
            f"{len(fold_folders)} folds"
        )


    # ==================================================
    # Process Folds
    # ==================================================

    for fold in fold_folders:

        fold_path = os.path.join(
            IMAGE_DATASET_PATH,
            fold
        )


        if DEBUG:

            print(
                f"\nProcessing : "
                f"{fold}"
            )


        # ==================================================
        # Get Pig Folders
        # ==================================================

        pig_folders = [
            folder
            for folder in os.listdir(
                fold_path
            )
            if os.path.isdir(
                os.path.join(
                    fold_path,
                    folder
                )
            )
        ]


        # Sort by Ground Truth Weight

        pig_folders = sorted(
            pig_folders,
            key=lambda x: float(
                x.split("_")[0]
            )
        )


        # ==================================================
        # Process Pig Folders
        # ==================================================

        for pig_folder in pig_folders:

            pig_folder_path = os.path.join(
                fold_path,
                pig_folder
            )


            if DEBUG:

                print(
                    f"  Processing Pig : "
                    f"{pig_folder}"
                )


            # ==================================================
            # Get Image Files
            # ==================================================

            image_files = sorted(
                [
                    file
                    for file in os.listdir(
                        pig_folder_path
                    )
                    if file.lower().endswith(
                        (
                            ".png",
                            ".jpg",
                            ".jpeg"
                        )
                    )
                ],
                key=lambda x: int(
                    os.path.splitext(x)[0]
                    .split("_")[-1]
                )
            )


            # ==================================================
            # Create Output Folders
            # ==================================================

            mask_output_folder = os.path.join(
                MASK_OUTPUT_FOLDER,
                fold,
                pig_folder
            )


            refined_mask_output_folder = os.path.join(
                REFINED_MASK_OUTPUT_FOLDER,
                fold,
                pig_folder
            )


            target_mask_output_folder = os.path.join(
                TARGET_MASK_OUTPUT_FOLDER,
                fold,
                pig_folder
            )


            review_output_folder = os.path.join(
                SEGMENTATION_REVIEW_FOLDER,
                fold,
                pig_folder
            )


            # ==================================================
            # Process Images
            # ==================================================

            if TEST_MODE:

                image_files = image_files[
                    :TEST_IMAGE_COUNT
                ]


            for image_file in image_files:

                total_images += 1

                # Image path

                image_path = os.path.join(
                    pig_folder_path,
                    image_file
                )


                # Image name without extension

                image_name = os.path.splitext(
                    image_file
                )[0]


                # ==================================================
                # Read Image
                # ==================================================

                image = read_image(
                    image_path
                )


                # ==================================================
                # Segment Pig
                # ==================================================

                masks, refined_masks = (
                    segment_pigs(
                        model,
                        image
                    )
                )


                # ==================================================
                # No Mask
                # ==================================================

                if len(masks) == 0:

                    no_mask_images += 1

                    print(
                        f"No mask : "
                        f"{fold}/{pig_folder}/"
                        f"{image_file}"
                    )

                    continue


                # ==================================================
                # No Refined Mask
                # ==================================================

                if len(refined_masks) == 0:

                    no_refined_mask_images += 1

                    print(
                        f"No refined mask : "
                        f"{fold}/{pig_folder}/"
                        f"{image_file}"
                    )

                    continue


                # ==================================================
                # Select Target Mask
                # ==================================================

                target_mask = (
                    select_target_mask(
                        refined_masks
                    )
                )


                # ==================================================
                # No Target Mask
                # ==================================================

                if target_mask is None:

                    no_target_mask_images += 1

                    print(
                        f"No target mask : "
                        f"{fold}/{pig_folder}/"
                        f"{image_file}"
                    )

                    continue


                # ==================================================
                # Save Target Mask
                # ==================================================

                save_masks(
                    [target_mask],
                    target_mask_output_folder,
                    image_name,
                    prefix="target"
                )


                # ==================================================
                # Save Original Masks
                # ==================================================

                save_masks(
                    masks,
                    mask_output_folder,
                    image_name,
                    prefix="mask"
                )


                # ==================================================
                # Save Refined Masks
                # ==================================================

                save_masks(
                    refined_masks,
                    refined_mask_output_folder,
                    image_name,
                    prefix="refined_mask"
                )


                # ==================================================
                # Create Review Image
                # ==================================================

                review_image = (
                    create_review_image(
                        image,
                        target_mask
                    )
                )


                # ==================================================
                # Save Review Image
                # ==================================================

                os.makedirs(
                    review_output_folder,
                    exist_ok=True
                )


                review_path = os.path.join(
                    review_output_folder,
                    f"{image_name}_review.jpg"
                )


                cv2.imwrite(
                    review_path,
                    review_image
                )

                processed_images += 1


                # ==================================================
                # Debug
                # ==================================================

                if DEBUG:

                    print(
                        f"    Processed : "
                        f"{image_file}"
                    )


    # ==================================================
    # Completion
    # ==================================================

    print()
    print("=" * 60)
    print("สรุปผลการทำ Segmentation")
    print("=" * 60)

    print(
        f"จำนวนภาพทั้งหมด          : {total_images:,} ภาพ"
    )

    print(
        f"ไม่พบ Mask                : {no_mask_images:,} ภาพ"
    )

    print(
        f"ไม่พบ Refined Mask        : {no_refined_mask_images:,} ภาพ"
    )

    print(
        f"ไม่พบ Target Mask         : {no_target_mask_images:,} ภาพ"
    )

    print(
        f"ประมวลผลสำเร็จ            : {processed_images:,} ภาพ"
    )

    print("=" * 60)
    print("Segmentation completed.")


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":

    main()