# ======================================================
# Import Libraries
# ======================================================

import os


# ======================================================
# Dataset Configuration
# ======================================================

DATASET_PATH = os.path.abspath(
    os.path.join(
        "dataset",
        "PigRGB-Weight",
        "transfer_learning_dataset"
    )
)

DATA_YAML_PATH = os.path.join(
    DATASET_PATH,
    "data.yaml"
)


# ======================================================
# Create data.yaml
# ======================================================

with open(
    DATA_YAML_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"path: {DATASET_PATH.replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: pig\n"
    )


# ======================================================
# Display Result
# ======================================================

print("=" * 60)
print("DATA YAML CREATION COMPLETED")
print("=" * 60)

print()

print(
    f"Dataset path : {DATASET_PATH}"
)

print(
    f"YAML path    : {DATA_YAML_PATH}"
)

print()

print(
    "Class:"
)

print(
    "0 : pig"
)

print()

print("=" * 60)
print("DATASET YAML CREATION COMPLETED")
print("=" * 60)