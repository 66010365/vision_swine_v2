# ======================================================
# Import Libraries
# ======================================================

import os
import random
import shutil


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

OUTPUT_DATASET_PATH = os.path.join(
    "dataset",
    "PigRGB-Weight",
    "transfer_learning_dataset"
)


# ======================================================
# Split Configuration
# ======================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# ======================================================
# Check Split Ratio
# ======================================================

if abs(
    TRAIN_RATIO
    + VAL_RATIO
    + TEST_RATIO
    - 1.0
) > 1e-6:

    raise ValueError(
        "Train / Val / Test ratios must equal 1.0"
    )


# ======================================================
# Create Output Folders
# ======================================================

for split in [
    "train",
    "val",
    "test"
]:

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET_PATH,
            "images",
            split
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET_PATH,
            "labels",
            split
        ),
        exist_ok=True
    )


# ======================================================
# Get Dataset Groups
# ======================================================

groups = []

for root, dirs, files in os.walk(
    RGB_DATASET_PATH
):

    image_files = [
        file
        for file in files
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    if len(image_files) > 0:

        group_name = os.path.basename(
            root
        )

        groups.append(
            {
                "name": group_name,
                "path": root,
                "images": sorted(
                    image_files
                )
            }
        )


# ======================================================
# Sort Groups
# ======================================================

groups = sorted(
    groups,
    key=lambda group: group["name"]
)


# ======================================================
# Display Dataset Information
# ======================================================

print("=" * 60)
print("GROUP DATASET SPLIT")
print("=" * 60)

print()

print(
    f"Total groups : {len(groups)}"
)

print(
    f"Total images : "
    f"{sum(len(g['images']) for g in groups)}"
)

print()

print(
    "Target Split"
)

print(
    f"Train : {TRAIN_RATIO * 100:.1f}%"
)

print(
    f"Val   : {VAL_RATIO * 100:.1f}%"
)

print(
    f"Test  : {TEST_RATIO * 100:.1f}%"
)


# ======================================================
# Shuffle Groups
# ======================================================

random.seed(
    RANDOM_STATE
)

random.shuffle(
    groups
)


# ======================================================
# Calculate Target Image Counts
# ======================================================

total_images = sum(
    len(group["images"])
    for group in groups
)

target_train = (
    total_images
    * TRAIN_RATIO
)

target_val = (
    total_images
    * VAL_RATIO
)

target_test = (
    total_images
    * TEST_RATIO
)


# ======================================================
# Assign Groups
# ======================================================

split_groups = {
    "train": [],
    "val": [],
    "test": []
}

split_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}


# ======================================================
# Greedy Group Assignment
# ======================================================

for group in groups:

    remaining_targets = {
        "train": target_train
        - split_counts["train"],

        "val": target_val
        - split_counts["val"],

        "test": target_test
        - split_counts["test"]
    }

    best_split = max(
        remaining_targets,
        key=remaining_targets.get
    )

    split_groups[
        best_split
    ].append(
        group
    )

    split_counts[
        best_split
    ] += len(
        group["images"]
    )


# ======================================================
# Display Group Assignment
# ======================================================

print()
print("=" * 60)
print("GROUP ASSIGNMENT")
print("=" * 60)

for split in [
    "train",
    "val",
    "test"
]:

    print()
    print(
        f"{split.upper()}"
    )

    print(
        "-" * 60
    )

    for group in split_groups[
        split
    ]:

        print(
            f"{group['name']:<20} "
            f"-> "
            f"{len(group['images'])} images"
        )


# ======================================================
# Copy Images and Labels
# ======================================================

print()
print("=" * 60)
print("COPYING DATASET")
print("=" * 60)

for split in [
    "train",
    "val",
    "test"
]:

    print()
    print(
        f"Processing {split}..."
    )

    for group in split_groups[
        split
    ]:

        for image_file in group[
            "images"
        ]:

            image_source = os.path.join(
                group["path"],
                image_file
            )

            image_name = os.path.splitext(
                image_file
            )[0]

            label_file = (
                image_name
                + ".txt"
            )

            label_source = os.path.join(
                LABEL_DATASET_PATH,
                label_file
            )

            # ==========================================
            # Check Label
            # ==========================================

            if not os.path.exists(
                label_source
            ):

                print(
                    f"Missing label : "
                    f"{image_name}"
                )

                continue

            # ==========================================
            # Destination
            # ==========================================

            image_destination = os.path.join(
                OUTPUT_DATASET_PATH,
                "images",
                split,
                image_file
            )

            label_destination = os.path.join(
                OUTPUT_DATASET_PATH,
                "labels",
                split,
                label_file
            )

            # ==========================================
            # Copy Image
            # ==========================================

            shutil.copy2(
                image_source,
                image_destination
            )

            # ==========================================
            # Copy Label
            # ==========================================

            shutil.copy2(
                label_source,
                label_destination
            )


# ======================================================
# Final Summary
# ======================================================

print()
print("=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

for split in [
    "train",
    "val",
    "test"
]:

    count = split_counts[
        split
    ]

    percentage = (
        count
        / total_images
        * 100
    )

    print(
        f"{split.upper():5s} : "
        f"{count:4d} images "
        f"({percentage:.2f}%)"
    )


# ======================================================
# Completion
# ======================================================

print()
print("=" * 60)
print("GROUP DATASET SPLIT COMPLETED")
print("=" * 60)

print(
    f"Output folder : "
    f"{OUTPUT_DATASET_PATH}"
)