# ============================================================
# Segmentation Quality Control
# ============================================================

import os
import re
import pandas as pd


# ============================================================
# Import Configuration
# ============================================================

from config import (
    IMAGE_DATASET_PATH,
    FAILED_IMAGES_PATH,
    USABLE_IMAGES_PATH,
    TARGET_MASK_OUTPUT_FOLDER,
)

# ============================================================
# File Extensions
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Name Normalization
# ============================================================

def get_rgb_name(file_name):
    """
    Get RGB image name without extension.

    Example:
        101.9kg_121.jpg
        ->
        101.9kg_121
    """

    return os.path.splitext(
        os.path.basename(str(file_name))
    )[0]


def get_mask_name(file_name):
    """
    Convert Target Mask filename
    to original RGB image name.

    Example:

        73.36kg_124_target_1.png

    becomes:

        73.36kg_124
    """

    base_name = os.path.splitext(
        os.path.basename(str(file_name))
    )[0]

    base_name = re.sub(
        r"_target_\d+$",
        "",
        base_name,
        flags=re.IGNORECASE
    )

    return base_name


def get_failed_name(file_name):
    """
    Get original RGB image name from failed-image filename.

    Examples:

        101.9kg_121_target_1.png
        101.9kg_121_target_2.png

    become:

        101.9kg_121
    """

    base_name = os.path.splitext(
        os.path.basename(str(file_name))
    )[0]

    base_name = re.sub(
        r"_target_\d+$",
        "",
        base_name,
        flags=re.IGNORECASE
    )

    return base_name


# ============================================================
# Get RGB Files
# ============================================================

def get_rgb_files(folder):
    """
    Get all RGB images recursively.

    Returns:

        {
            "image_name": "full/path/to/image.jpg"
        }
    """

    rgb_files = {}

    for root, dirs, files in os.walk(folder):

        for file_name in files:

            if not file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            image_name = get_rgb_name(
                file_name
            )

            full_path = os.path.join(
                root,
                file_name
            )

            rgb_files[image_name] = full_path

    return rgb_files


# ============================================================
# Get Mask Files
# ============================================================

def get_mask_files(folder):
    """
    Get all Target Mask files recursively.

    Example:

        image_001_mask_1.png
        image_001_mask_2.png

    becomes:

        {
            "image_001": [
                ".../image_001_mask_1.png",
                ".../image_001_mask_2.png"
            ]
        }
    """

    mask_files = {}

    if not os.path.exists(folder):

        return mask_files

    for root, dirs, files in os.walk(folder):

        for file_name in files:

            if not file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            mask_name = get_mask_name(
                file_name
            )

            full_path = os.path.join(
                root,
                file_name
            )

            if mask_name not in mask_files:

                mask_files[mask_name] = []

            mask_files[mask_name].append(
                full_path
            )

    return mask_files


# ============================================================
# Get Failed Files
# ============================================================

def get_failed_names(folder):
    """
    Get normalized failed-image names.
    """

    failed_names = set()

    if not os.path.exists(folder):

        return failed_names

    for root, dirs, files in os.walk(folder):

        for file_name in files:

            if not file_name.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            failed_name = get_failed_name(
                file_name
            )

            failed_names.add(
                failed_name
            )

    return failed_names


# ============================================================
# Create Usable Images Excel
# ============================================================

def create_usable_images_excel(
    usable_names,
    rgb_files
):
    """
    Create usable_images.xlsx
    from the usable image names.

    Usable image:
        RGB + Target Mask - Failed
    """

    rows = []

    for image_name in sorted(usable_names):

        rows.append({
            "No.": len(rows) + 1,
            "Image Name": image_name,
            "Image Path": rgb_files[image_name]
        })

    df_usable = pd.DataFrame(rows)

    # Create parent folder if needed
    usable_folder = os.path.dirname(
        USABLE_IMAGES_PATH
    )

    if usable_folder:
        os.makedirs(
            usable_folder,
            exist_ok=True
        )

    df_usable.to_excel(
        USABLE_IMAGES_PATH,
        index=False
    )

    print("\n[สร้างไฟล์ชุดข้อมูลที่สามารถใช้งานได้]")

    print(
        f"บันทึกไฟล์ไว้ที่:"
        f"\n{USABLE_IMAGES_PATH}"
    )

    print(
        f"จำนวนภาพ      : "
        f"{len(df_usable):,}"
    )


# ============================================================
# 1. ตรวจสอบภาพรวมของภาพผิดหนักและ Target Mask
# ============================================================

def check_failed_overlap():
    """
    Check:

        RGB
        Mask
        Failed images
        Usable images
    """

    print("\n" + "=" * 60)
    print("1. ตรวจสอบภาพรวมของภาพผิดหนักและ Target Mask")
    print("=" * 60)

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_files = get_rgb_files(
        IMAGE_DATASET_PATH
    )

    rgb_names = set(
        rgb_files.keys()
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    target_mask_files = get_mask_files(
        TARGET_MASK_OUTPUT_FOLDER
    )

    target_mask_names = set(
        target_mask_files.keys()
    )

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    failed_names = get_failed_names(
        FAILED_IMAGES_PATH
    )

    # --------------------------------------------------------
    # RGB without Mask
    # --------------------------------------------------------

    rgb_no_mask = (
        rgb_names - target_mask_names
    )

    # --------------------------------------------------------
    # Failed images that exist in RGB
    # --------------------------------------------------------

    failed_in_rgb = (
        failed_names & rgb_names
    )

    # --------------------------------------------------------
    # Failed images that also have no Mask
    # --------------------------------------------------------

    failed_no_mask = (
        failed_names & rgb_no_mask
    )

    # --------------------------------------------------------
    # Usable images
    #
    # RGB ∩ Mask - Failed
    # --------------------------------------------------------

    usable_names = (
        (rgb_names & target_mask_names)
        - failed_names
    )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    total_rgb = len(
        rgb_names
    )

    total_mask_names = len(
        target_mask_names
    )

    total_failed = len(
        failed_names
    )

    total_rgb_no_mask = len(
        rgb_no_mask
    )

    total_failed_in_rgb = len(
        failed_in_rgb
    )

    total_failed_no_mask = len(
        failed_no_mask
    )

    total_usable = len(
        usable_names
    )

    usable_percentage = (
        total_usable / total_rgb * 100
        if total_rgb > 0
        else 0
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nจำนวนภาพ RGB ทั้งหมด          : "
        f"{total_rgb:,}"
    )

    print(
        f"จำนวน Target Mask             : "
        f"{total_mask_names:,}"
    )

    print(
        f"ภาพ RGB ไม่มี Target Mask     : "
        f"{total_rgb_no_mask:,}"
    )

    print(
        f"\nจำนวนภาพผิดหนัก               : "
        f"{total_failed:,}"
    )

    print(
        f"ภาพผิดหนักที่มี Target Mask   : "
        f"{total_failed_in_rgb - total_failed_no_mask:,}"
    )

    print(
        f"ภาพผิดหนักที่ไม่มี Target Mask: "
        f"{total_failed_no_mask:,}"
    )

    print(
        f"\nจำนวนภาพที่คำนวณว่าใช้ได้      : "
        f"{total_usable:,}"
    )

    print(
        f"อัตราภาพที่สามารถใช้งานได้     : "
        f"{usable_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_unusable = total_rgb - total_usable

    print("\n[สรุป]")

    print(
        f"ภาพที่สามารถนำไปใช้ต่อ         : "
        f"{total_usable:,} ภาพ"
    )

    print(
        f"ภาพที่ไม่สามารถนำไปใช้ต่อ      : "
        f"{total_unusable:,} ภาพ"
    )

    # --------------------------------------------------------
    # Create usable_images.xlsx
    # --------------------------------------------------------

    create_usable_images_excel(
        usable_names,
        rgb_files
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "rgb_names": rgb_names,
        "mask_names": target_mask_names,
        "failed_names": failed_names,
        "usable_names": usable_names,
        "mask_files": target_mask_files
    }


# ============================================================
# 2. CHECK USABLE DATASET
# ============================================================

def check_usable_dataset(usable_names):
    """
    Compare the usable_images.xlsx dataset
    with the usable images calculated directly
    from RGB + Target Mask - Failed.
    """

    print("\n" + "=" * 60)
    print("2. ตรวจสอบชุดข้อมูลที่สามารถใช้งานได้")
    print("=" * 60)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(
        USABLE_IMAGES_PATH
    ):

        print(
            "\n[ERROR] usable_images.xlsx "
            "not found."
        )

        return

    # --------------------------------------------------------
    # Load Excel
    # --------------------------------------------------------

    df = pd.read_excel(
        USABLE_IMAGES_PATH
    )

    # --------------------------------------------------------
    # Check column
    # --------------------------------------------------------

    if "Image Name" not in df.columns:

        print(
            "\n[ERROR] "
            "ไม่พบคอลัมน์ 'Image Name'"
        )

        print(
            "\nคอลัมน์ที่มีอยู่:"
        )

        for column in df.columns:

            print(
                f"    {column}"
            )

        return

    # --------------------------------------------------------
    # Excel names
    # --------------------------------------------------------

    excel_names = set(
        df["Image Name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    only_in_excel = (
        excel_names - usable_names
    )

    only_in_calculated = (
        usable_names - excel_names
)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nจำนวนแถวใน usable_images.xlsx : "
        f"{len(df)}"
    )

    print(
        f"จำนวนชื่อภาพไม่ซ้ำใน Excel     : "
        f"{len(excel_names)}"
    )

    print(
        f"จำนวนที่คำนวณได้         : "
        f"{len(usable_names)}"
    )

    print(
        f"มีเฉพาะใน Excel         : "
        f"{len(only_in_excel)}"
    )

    print(
        f"มีเฉพาะในผลคำนวณ       : "
        f"{len(only_in_calculated)}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if (
        len(only_in_excel) == 0
        and len(only_in_calculated) == 0
    ):

        print(
            "\n[ผ่าน] ข้อมูลใน usable_images.xlsx "
            "ตรงกับชุดข้อมูลที่คำนวณได้"
        )

    else:

        print(
            "\n[คำเตือน] ข้อมูลใน usable_images.xlsx "
            "ไม่ตรงกับชุดข้อมูลที่คำนวณได้"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("การตรวจสอบคุณภาพการแบ่งส่วนภาพ")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Failed image check
    # --------------------------------------------------------

    qc_result = check_failed_overlap()

    # --------------------------------------------------------
    # 2. Check usable dataset
    # --------------------------------------------------------

    check_usable_dataset(
        qc_result["usable_names"]
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("การตรวจสอบคุณภาพการแบ่งส่วนภาพเสร็จสิ้น")
    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()