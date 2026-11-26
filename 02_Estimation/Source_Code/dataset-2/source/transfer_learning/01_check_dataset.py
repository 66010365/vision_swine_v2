# ======================================================
# Import Libraries
# ======================================================

import os
from PIL import Image


# ======================================================
# Dataset Paths
# ======================================================

RGB_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "RGB_MASK_3394",
    "RGB_3394"
)

MASK_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "RGB_MASK_3394",
    "MASK_3394"
)


# ======================================================
# Dataset Configuration
# ======================================================

EXPECTED_TOTAL_IMAGES = 3394


# ======================================================
# Supported Image Extensions
# ======================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ======================================================
# Get All Images Recursively
# ======================================================

def get_image_files(folder):

    image_files = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                image_files.append(
                    os.path.join(root, file)
                )

    return sorted(image_files)


# ======================================================
# Create File Key
# ======================================================

def create_key(path):

    filename = os.path.splitext(
        os.path.basename(path)
    )[0]

    return filename.lower()


# ======================================================
# Check Image
# ======================================================

def check_image(path):

    try:

        image = Image.open(path)

        width, height = image.size

        return True, width, height

    except Exception:

        return False, None, None


# ======================================================
# Check Dataset Groups
# ======================================================

def get_dataset_groups(folder):

    groups = []

    for root, dirs, files in os.walk(folder):

        image_files = [
            file
            for file in files
            if file.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ]

        if len(image_files) > 0:

            groups.append(
                (
                    root,
                    sorted(image_files)
                )
            )

    return groups


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 60)
    print("PIGRGB-WEIGHT DATASET CHECK")
    print("=" * 60)


    # ==================================================
    # Check Dataset Paths
    # ==================================================

    if not os.path.exists(RGB_DATASET_PATH):

        print("\nERROR: RGB dataset folder not found:")
        print(RGB_DATASET_PATH)

        return


    if not os.path.exists(MASK_DATASET_PATH):

        print("\nERROR: Mask dataset folder not found:")
        print(MASK_DATASET_PATH)

        return


    # ==================================================
    # Get RGB Images
    # ==================================================

    rgb_files = get_image_files(
        RGB_DATASET_PATH
    )


    # ==================================================
    # Get Mask Images
    # ==================================================

    mask_files = get_image_files(
        MASK_DATASET_PATH
    )


    # ==================================================
    # Display Dataset Counts
    # ==================================================

    print("\nDataset")
    print("-" * 60)

    print(
        f"RGB images  : {len(rgb_files)}"
    )

    print(
        f"Mask images : {len(mask_files)}"
    )


    # ==================================================
    # Create Dictionaries
    # ==================================================

    rgb_dict = {
        create_key(path): path
        for path in rgb_files
    }

    mask_dict = {
        create_key(path): path
        for path in mask_files
    }


    # ==================================================
    # Find Matching Files
    # ==================================================

    rgb_keys = set(rgb_dict.keys())
    mask_keys = set(mask_dict.keys())

    matched_keys = (
        rgb_keys & mask_keys
    )

    missing_mask = (
        rgb_keys - mask_keys
    )

    missing_rgb = (
        mask_keys - rgb_keys
    )


    # ==================================================
    # Display Matching Results
    # ==================================================

    print("\nMatching")
    print("-" * 60)

    print(
        f"Matched pairs : {len(matched_keys)}"
    )

    print(
        f"Missing mask  : {len(missing_mask)}"
    )

    print(
        f"Missing RGB   : {len(missing_rgb)}"
    )


    # ==================================================
    # Check Image Sizes
    # ==================================================

    size_mismatch = []

    invalid_rgb = []
    invalid_mask = []

    print("\nChecking image sizes...")


    for key in matched_keys:

        rgb_path = rgb_dict[key]
        mask_path = mask_dict[key]


        # ----------------------------------------------
        # Check RGB
        # ----------------------------------------------

        rgb_valid, rgb_width, rgb_height = (
            check_image(rgb_path)
        )

        if not rgb_valid:

            invalid_rgb.append(rgb_path)

            continue


        # ----------------------------------------------
        # Check Mask
        # ----------------------------------------------

        mask_valid, mask_width, mask_height = (
            check_image(mask_path)
        )

        if not mask_valid:

            invalid_mask.append(mask_path)

            continue


        # ----------------------------------------------
        # Compare Size
        # ----------------------------------------------

        if (
            rgb_width != mask_width
            or
            rgb_height != mask_height
        ):

            size_mismatch.append(
                (
                    key,
                    rgb_width,
                    rgb_height,
                    mask_width,
                    mask_height
                )
            )


    # ==================================================
    # Display Size Results
    # ==================================================

    print("\nImage Size Check")
    print("-" * 60)

    print(
        f"Size mismatch : {len(size_mismatch)}"
    )

    print(
        f"Invalid RGB   : {len(invalid_rgb)}"
    )

    print(
        f"Invalid Mask  : {len(invalid_mask)}"
    )


    # ==================================================
    # Dataset Group Check
    # ==================================================

    groups = get_dataset_groups(
        MASK_DATASET_PATH
    )

    total_images = sum(
        len(files)
        for _, files in groups
    )


    print("\nDataset Groups")
    print("-" * 60)

    print(
        f"Total groups : {len(groups)}"
    )

    print(
        f"Total images : {total_images}"
    )


    # ==================================================
    # Display Groups
    # ==================================================

    for index, (group_path, files) in enumerate(
        groups,
        start=1
    ):

        group_name = os.path.basename(
            group_path
        )

        print(
            f"Group {index:3d} : "
            f"{group_name:<20} -> "
            f"{len(files):3d} images"
        )


    # ==================================================
    # Display Missing Mask
    # ==================================================

    if missing_mask:

        print("\nRGB without Mask")
        print("-" * 60)

        for key in sorted(missing_mask):

            print(
                key,
                "->",
                rgb_dict[key]
            )


    # ==================================================
    # Display Missing RGB
    # ==================================================

    if missing_rgb:

        print("\nMask without RGB")
        print("-" * 60)

        for key in sorted(missing_rgb):

            print(
                key,
                "->",
                mask_dict[key]
            )


    # ==================================================
    # Display Size Mismatch
    # ==================================================

    if size_mismatch:

        print("\nSize Mismatch")
        print("-" * 60)

        for item in size_mismatch:

            (
                key,
                rgb_width,
                rgb_height,
                mask_width,
                mask_height
            ) = item

            print(
                f"{key}: "
                f"RGB={rgb_width}x{rgb_height}, "
                f"Mask={mask_width}x{mask_height}"
            )


    # ==================================================
    # Final Result
    # ==================================================

    print("\n" + "=" * 60)


    if (
        len(rgb_files) == EXPECTED_TOTAL_IMAGES
        and
        len(mask_files) == EXPECTED_TOTAL_IMAGES
        and
        len(missing_mask) == 0
        and
        len(missing_rgb) == 0
        and
        len(size_mismatch) == 0
        and
        len(invalid_rgb) == 0
        and
        len(invalid_mask) == 0
        and
        total_images == EXPECTED_TOTAL_IMAGES
    ):

        print(
            "DATASET CHECK PASSED"
        )

        print(
            "RGB and Mask datasets are ready."
        )

    else:

        print(
            "DATASET CHECK FOUND PROBLEMS"
        )

        print(
            "Please check the results above."
        )


    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":

    main()