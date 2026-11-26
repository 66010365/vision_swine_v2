# ======================================================
# Import Libraries
# ======================================================

import pandas as pd


# ======================================================
# Import Configuration
# ======================================================

from config import (
    FEATURE_DATASET_PATH,
    WEIGHT_GROUND_TRUTH_PATH,
    MODEL_DATASET_PATH,
    DEBUG
)


# ======================================================
# Normalize Image Name
# ======================================================

def normalize_image_name(image_name):

    image_name = str(image_name).strip()

    # Remove file extensions
    image_name = image_name.replace(".png", "")
    image_name = image_name.replace(".jpg", "")
    image_name = image_name.replace(".jpeg", "")

    # Remove target suffix if present
    image_name = image_name.replace("_target_1", "")

    return image_name


# ======================================================
# Validate Model Dataset
# ======================================================

def validate_model_dataset(
    model_df,
    feature_df,
    raw_df,
    missing_ground_truth
):

    print()
    print("=" * 60)
    print("การตรวจสอบชุดข้อมูล")
    print("=" * 60)

    # ==================================================
    # 1. Basic Information
    # ==================================================

    total_images = len(model_df)
    total_pig_id = model_df["Pig_ID"].nunique()

    print()
    print(
        f"จำนวนภาพทั้งหมด          : {total_images:,}"
    )

    print(
        f"จำนวนสุกรทั้งหมด          : {total_pig_id:,}"
    )


    # ==================================================
    # 2. Check Required Columns
    # ==================================================

    required_columns = [
        "Image",
        "Image_Path",
        "Mask_Path",
        "Area",
        "Perimeter",
        "Bounding_Box_Length",
        "Bounding_Box_Width",
        "Deviation",
        "Skeleton_Length",
        "Straight_Distance",
        "Straightness",
        "Pig_ID",
        "Weight"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in model_df.columns
    ]


    # ==================================================
    # 3. Check Missing Values
    # ==================================================

    check_columns = [
        "Pig_ID",
        "Weight",
        "Area",
        "Perimeter",
        "Bounding_Box_Length",
        "Bounding_Box_Width",
        "Deviation",
        "Skeleton_Length",
        "Straight_Distance",
        "Straightness"
    ]

    missing_values = (
        model_df[
            check_columns
        ]
        .isnull()
        .sum()
    )

    total_missing_values = (
        missing_values.sum()
    )


    # ==================================================
    # 4. Check Duplicate Images
    # ==================================================

    duplicate_images = (
        model_df["Image"]
        .duplicated()
        .sum()
    )


    # ==================================================
    # 5. Check Ground Truth Matching
    # ==================================================

    matched_images = (
        len(feature_df)
        - len(missing_ground_truth)
    )

    missing_ground_truth_count = (
        len(missing_ground_truth)
    )


    # ==================================================
    # 6. Check Weight per Pig
    # ==================================================

    weight_check = (
        model_df
        .groupby("Pig_ID")["Weight"]
        .agg(
            [
                "count",
                "min",
                "max",
                "mean",
                "std"
            ]
        )
    )

    # std can be NaN when a Pig has only 1 image
    weight_std = (
        weight_check["std"]
        .fillna(0)
    )

    inconsistent_weight = (
        weight_std > 0
    ).sum()


    # ==================================================
    # 7. Print Validation Results
    # ==================================================

    print()
    print("ผลการตรวจสอบ")
    print("-" * 60)

    print(
        f"{'ภาพที่จับคู่ Ground Truth ได้':<28} : {matched_images:,} ภาพ"
    )

    print(
        f"{'ภาพที่ไม่มี Ground Truth':<28} : {missing_ground_truth_count:,}"
    )

    print(
        f"{'น้ำหนักไม่สอดคล้องกัน':<28} : {inconsistent_weight:,}"
    )

    print(
        f"{'คอลัมน์ที่จำเป็นขาดหาย':<28} : {len(missing_columns)}"
    )

    print(
        f"{'ข้อมูลที่ขาดหาย':<28} : {total_missing_values:,}"
    )

    print(
        f"{'ภาพซ้ำ':<28} : {duplicate_images:,}"
    )

    # ==================================================
    # 9. Determine PASS / FAIL
    # ==================================================

    validation_pass = True

    if len(missing_columns) > 0:
        validation_pass = False

    if total_missing_values > 0:
        validation_pass = False

    if duplicate_images > 0:
        validation_pass = False

    if missing_ground_truth_count > 0:
        validation_pass = False

    if inconsistent_weight > 0:
        validation_pass = False


    # ==================================================
    # 10. Final Validation Result
    # ==================================================

    print()
    print("=" * 60)

    if validation_pass:

        print(
            "ผลการตรวจสอบชุดข้อมูล : ผ่าน"
        )

    else:

        print(
            "ผลการตรวจสอบชุดข้อมูล : ไม่ผ่าน"
        )

    print("=" * 60)


    return validation_pass


# ======================================================
# Main
# ======================================================

def main():

    # ==================================================
    # Read Datasets
    # ==================================================

    feature_df = pd.read_excel(
        FEATURE_DATASET_PATH
    )

    raw_df = pd.read_excel(
        WEIGHT_GROUND_TRUTH_PATH
    )


    # ==================================================
    # Dataset Information
    # ==================================================

    print("=" * 60)
    print("การรวมชุดข้อมูล")
    print("=" * 60)


    # ==================================================
    # Normalize Image Names
    # ==================================================

    feature_df["Image"] = (
        feature_df["Image"]
        .apply(normalize_image_name)
    )

    raw_df["Image"] = (
        raw_df["Image"]
        .apply(normalize_image_name)
    )


    # ==================================================
    # Check Duplicate Image Names
    # ==================================================

    feature_duplicates = (
        feature_df["Image"]
        .duplicated()
        .sum()
    )

    raw_duplicates = (
        raw_df["Image"]
        .duplicated()
        .sum()
    )

    print()


    # ==================================================
    # Merge Datasets
    # ==================================================

    model_df = pd.merge(
        feature_df,
        raw_df,
        on="Image",
        how="inner"
    )


    # ==================================================
    # Check Missing Ground Truth
    # ==================================================

    feature_images = set(
        feature_df["Image"]
    )

    raw_images = set(
        raw_df["Image"]
    )

    matched_images = (
        feature_images
        & raw_images
    )

    missing_ground_truth = (
        feature_images
        - raw_images
    )


    # ==================================================
    # Check Required Ground Truth Columns
    # ==================================================

    if "Weight" not in model_df.columns:

        raise ValueError(
            "Weight column was not found "
            "in the Ground Truth dataset."
        )


    if "Pig_ID" not in model_df.columns:

        raise ValueError(
            "Pig_ID column was not found "
            "in the Ground Truth dataset."
        )


    # ==================================================
    # Validate Dataset
    # ==================================================

    validation_pass = validate_model_dataset(
        model_df,
        feature_df,
        raw_df,
        missing_ground_truth
    )


    # ==================================================
    # Save Model Dataset
    # ==================================================

    model_df.to_excel(
        MODEL_DATASET_PATH,
        index=False
    )


    # ==================================================
    # Final Result
    # ==================================================

    print()
    print("=" * 60)
    print("สรุปผลการรวมชุดข้อมูล")
    print("=" * 60)

    print(
        f"Ground Truth    : {len(raw_df):,}"
    )

    print(
        f"Feature Dataset : {len(feature_df):,}"
    )

    print(
        f"Model Dataset   : {len(model_df):,}"
    )

    print()
    print(
        f"บันทึกไฟล์ไว้ที่ : {MODEL_DATASET_PATH}"
    )

    if validation_pass:

        print(
            "สถานะสุดท้าย   : พร้อมเข้าสู่การสร้างโมเดล"
        )

    else:

        print(
            "สถานะสุดท้าย   : กรุณาตรวจสอบชุดข้อมูล"
        )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()