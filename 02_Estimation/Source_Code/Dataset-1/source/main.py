# ======================================================
# Import Libraries
# ======================================================

import os

from segmentation import (
    load_model,
    read_image,
    segment_pigs,
    select_target_mask,
    save_masks
)

# ======================================================
# Define Paths
# ======================================================

from config import (
    YOLO_MODEL_PATH,
    IMAGE_FOLDER,
    MASK_OUTPUT_FOLDER,
    REFINED_MASK_OUTPUT_FOLDER,
    TARGET_MASK_OUTPUT_FOLDER,
    DEBUG
)

# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Load YOLO Model
    # ======================================================
    
    model = load_model(YOLO_MODEL_PATH)

    # ======================================================
    # Get Image Files
    # ======================================================
    
    image_files = sorted(
        [
            file
            for file in os.listdir(IMAGE_FOLDER)
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]
    )

    if DEBUG:
        print(f"Found {len(image_files)} images")

    # ======================================================
    # Process Images
    # ======================================================
    
    for image_file in image_files:

        # Image path
        image_path = os.path.join(
            IMAGE_FOLDER,
            image_file
        )

        # Image name (without extension)
        image_name = os.path.splitext(image_file)[0]

        # Read image
        image = read_image(image_path)

        # Segment pigs
        masks, refined_masks = segment_pigs(
            model,
            image
        )

        # Select target pig
        target_mask = select_target_mask(refined_masks)

        # No target
        if target_mask is None:
            print(f"No target mask : {image_file}")
            continue

        # Save target mask
        save_masks(
            [target_mask],
            TARGET_MASK_OUTPUT_FOLDER,
            image_name,
            prefix="target"
        )

        # Save original masks
        save_masks(
            masks,
            MASK_OUTPUT_FOLDER,
            image_name,
            prefix="mask"
        )

        # Save refined masks
        save_masks(
            refined_masks,
            REFINED_MASK_OUTPUT_FOLDER,
            image_name,
            prefix="refined_mask"
        )

        if DEBUG:
            print(f"Processed : {image_file}")

    # ======================================================
    # Completion
    # ======================================================
    
    print("Segmentation completed.")

# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()


