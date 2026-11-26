# ======================================================
# Import Libraries
# ======================================================

import os


# ======================================================
# Dataset Configuration
# ======================================================

DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "transfer_learning_dataset"
)


# ======================================================
# Split Names
# ======================================================

SPLITS = [
    "train",
    "val",
    "test"
]


# ======================================================
# Check Dataset Split
# ======================================================

def check_split(split):

    image_folder = os.path.join(
        DATASET_PATH,
        "images",
        split
    )

    label_folder = os.path.join(
        DATASET_PATH,
        "labels",
        split
    )

    # ==================================================
    # Get Images
    # ==================================================

    image_files = []

    for file in os.listdir(
        image_folder
    ):

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            image_files.append(
                file
            )

    # ==================================================
    # Get Labels
    # ==================================================

    label_files = []

    for file in os.listdir(
        label_folder
    ):

        if file.lower().endswith(
            ".txt"
        ):

            label_files.append(
                file
            )

    # ==================================================
    # Create File Names
    # ==================================================

    image_names = {
        os.path.splitext(file)[0]
        for file in image_files
    }

    label_names = {
        os.path.splitext(file)[0]
        for file in label_files
    }

    # ==================================================
    # Compare Images and Labels
    # ==================================================

    missing_labels = (
        image_names - label_names
    )

    missing_images = (
        label_names - image_names
    )

    # ==================================================
    # Check Empty Labels
    # ==================================================

    empty_labels = []

    for label_file in label_files:

        label_path = os.path.join(
            label_folder,
            label_file
        )

        if os.path.getsize(
            label_path
        ) == 0:

            empty_labels.append(
                label_file
            )

    # ==================================================
    # Display Result
    # ==================================================

    print(
        f"{split.upper()}"
    )

    print(
        "-" * 60
    )

    print(
        f"Images          : "
        f"{len(image_files)}"
    )

    print(
        f"Labels          : "
        f"{len(label_files)}"
    )

    print(
        f"Missing labels  : "
        f"{len(missing_labels)}"
    )

    print(
        f"Missing images  : "
        f"{len(missing_images)}"
    )

    print(
        f"Empty labels    : "
        f"{len(empty_labels)}"
    )

    print()

    # ==================================================
    # Display Problems
    # ==================================================

    if missing_labels:

        print(
            "Missing Labels:"
        )

        for name in sorted(
            missing_labels
        ):

            print(
                f"  {name}"
            )

        print()

    if missing_images:

        print(
            "Missing Images:"
        )

        for name in sorted(
            missing_images
        ):

            print(
                f"  {name}"
            )

        print()

    if empty_labels:

        print(
            "Empty Labels:"
        )

        for name in sorted(
            empty_labels
        ):

            print(
                f"  {name}"
            )

        print()

    # ==================================================
    # Return Status
    # ==================================================

    return (
        len(missing_labels) == 0
        and
        len(missing_images) == 0
        and
        len(empty_labels) == 0
    )


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 60)
    print("DATASET SPLIT CHECK")
    print("=" * 60)

    print()

    all_passed = True

    # ==================================================
    # Check All Splits
    # ==================================================

    for split in SPLITS:

        result = check_split(
            split
        )

        if not result:

            all_passed = False

    # ==================================================
    # Final Result
    # ==================================================

    print("=" * 60)

    if all_passed:

        print(
            "DATASET SPLIT CHECK PASSED"
        )

    else:

        print(
            "DATASET SPLIT CHECK FOUND PROBLEMS"
        )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":

    main()