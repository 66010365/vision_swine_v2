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
    YOLO_CONFIDENCE,
    MIN_MASK_AREA,
    BORDER_MARGIN,
    AREA_SCORE_SCALE,
    MAX_AREA_SCORE,
    COMPLETENESS_THRESHOLD,
    DEBUG
)

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
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image

# ======================================================
# Segment Image
# ======================================================

def segment_image(model, image):

    results = model(
        image,
        conf=YOLO_CONFIDENCE
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
        mask = (mask * 255).astype(np.uint8)
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
        print(f"Number of contours : {len(contours)}")

        for i, contour in enumerate(contours):
            print(f"Contour {i}: Area = {cv2.contourArea(contour)}")

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create empty mask
    refined_mask = np.zeros(mask.shape, dtype=np.uint8)

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

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return 0

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

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

    x, y, w, h = cv2.boundingRect(largest_contour)

    if (
        x <= BORDER_MARGIN or
        y <= BORDER_MARGIN or
        (x + w) >= (width - BORDER_MARGIN) or
        (y + h) >= (height - BORDER_MARGIN)
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

        if fill_ratio >= COMPLETENESS_THRESHOLD:
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
    
    center_score = (1 - distance / max_distance) * 30

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
        print(f"Area         : {area}")
        print(f"Center       : ({cx}, {cy})")
        print(f"Area Score   : {area_score:.2f}")
        print(f"Center Score : {center_score:.2f}")
        print(f"Border Score : {border_score:.2f}")
        print(f"Completeness : {completeness_score:.2f}")
        print(f"Total Score  : {score:.2f}")
    
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
            print(f"Mask {i+1} Score : {score:.2f}")

        if score > best_score:
            best_score = score
            best_mask = mask

    if DEBUG:
        print(f"Selected Score : {best_score:.2f}")

    return best_mask

# ======================================================
# Save Masks
# ======================================================

def save_masks(masks, output_folder, image_name, prefix="mask"):

    os.makedirs(output_folder, exist_ok=True)

    for i, mask in enumerate(masks):

        # Convert binary mask (0/1) -> (0/255)
        if mask.max() <= 1:
            mask_to_save = (mask * 255).astype(np.uint8)
        else:
            mask_to_save = mask.astype(np.uint8)

        file_name = os.path.join(
            output_folder,
            f"{image_name}_{prefix}_{i+1}.png"
        )

        cv2.imwrite(file_name, mask_to_save)

# ======================================================
# Segment Pigs
# ======================================================

def segment_pigs(model, image):

    # Step 1 : Segment image using YOLO
    results = segment_image(model, image)

    # Step 2 : Get all pig masks
    masks = get_pig_masks(results)

    # Step 3 : Refine all masks
    refined_masks = refine_masks(masks)

    return masks, refined_masks
