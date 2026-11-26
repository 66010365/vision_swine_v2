# ======================================================
# Import Libraries
# ======================================================

import pandas as pd

# ======================================================
# Import Configuration
# ======================================================

from config import (
    FEATURE_DATASET_PATH,
    RAW_DATASET_PATH,
    MODEL_DATASET_PATH,
    DEBUG
)

# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Read Datasets
    # ======================================================

    # Read feature dataset
    feature_df = pd.read_excel(
        FEATURE_DATASET_PATH
    )

    # Read raw dataset
    raw_df = pd.read_excel(
        RAW_DATASET_PATH
    )

    # Remove "_target_1" from image name
    feature_df["Image"] = (
        feature_df["Image"]
        .str.replace("_target_1", "", regex=False)
    )
    if DEBUG:
        print(feature_df["Image"].head())

    if DEBUG:
        print(raw_df.head())

    # ======================================================
    # Merge Datasets
    # ======================================================

    model_df = pd.merge(
        feature_df,
        raw_df,
        on="Image",
        how="inner"
    )
    if DEBUG:
        print(model_df.head())

    if DEBUG:
        print(model_df.columns)

    # ======================================================
    # Save Model Dataset
    # ======================================================

    model_df.to_excel(
        MODEL_DATASET_PATH,
        index=False
    )

    print(f"Feature Dataset : {len(feature_df)}")
    print(f"Raw Dataset     : {len(raw_df)}")
    print(f"Model Dataset   : {len(model_df)}")

    print("Dataset merging completed successfully.")

# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()