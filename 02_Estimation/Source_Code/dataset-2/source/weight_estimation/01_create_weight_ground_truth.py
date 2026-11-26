# ============================================================
# Import Libraries
# ============================================================

import os
import re
import pandas as pd

from config import (
    IMAGE_DATASET_PATH,
    WEIGHT_GROUND_TRUTH_PATH
)


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Initialize Data
    # ========================================================

    data = []

    pig_number = 1


    # ========================================================
    # Get Fold Folders
    # ========================================================

    fold_folders = sorted(
        [
            folder
            for folder in os.listdir(IMAGE_DATASET_PATH)
            if os.path.isdir(
                os.path.join(
                    IMAGE_DATASET_PATH,
                    folder
                )
            )
        ]
    )


    # ========================================================
    # Process Folds
    # ========================================================

    for fold in fold_folders:

        fold_path = os.path.join(
            IMAGE_DATASET_PATH,
            fold
        )


        # ====================================================
        # Get Pig Folders
        # ====================================================

        pig_folders = [
            folder
            for folder in os.listdir(fold_path)
            if os.path.isdir(
                os.path.join(
                    fold_path,
                    folder
                )
            )
        ]


        # ====================================================
        # Sort Pig Folders by Weight
        # ====================================================

        pig_folders = sorted(
            pig_folders,
            key=lambda x: float(
                x.split("_")[0]
            )
        )


        # ====================================================
        # Process Pig Folders
        # ====================================================

        for pig_folder in pig_folders:

            pig_folder_path = os.path.join(
                fold_path,
                pig_folder
            )


            # =================================================
            # Get Ground Truth Weight
            # =================================================

            try:

                weight = float(
                    pig_folder.split("_")[0]
                )

            except ValueError:

                print(
                    f"Skip folder : {pig_folder}"
                )

                continue


            # =================================================
            # Create Pig ID
            # =================================================

            pig_id = f"pig{pig_number:03d}"


            # =================================================
            # Get Image Files
            # =================================================

            image_files = sorted(
                [
                    file
                    for file in os.listdir(
                        pig_folder_path
                    )
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    )
                ],
                key=natural_sort_key
            )


            # =================================================
            # Record Ground Truth
            # =================================================

            for image_file in image_files:

                data.append({

                    "Pig_ID": pig_id,

                    "Image": image_file,

                    "Weight": weight

                })


            # =================================================
            # Next Pig
            # =================================================

            pig_number += 1


    # ========================================================
    # Create DataFrame
    # ========================================================

    df = pd.DataFrame(data)


    # ========================================================
    # Save Excel
    # ========================================================

    df.to_excel(
        WEIGHT_GROUND_TRUTH_PATH,
        index=False
    )


    # ========================================================
    # Completion
    # ========================================================

    print(
        "สร้างชุดข้อมูล Ground Truth น้ำหนักเรียบร้อยแล้ว"
    )

    print(
        f"จำนวนสุกร : "
        f"{df['Pig_ID'].nunique()} ตัว"
    )

    print(
        f"จำนวนภาพ : "
        f"{len(df):,} ภาพ"
    )

    print(
        f"บันทึกไฟล์ไว้ที่ : "
        f"{WEIGHT_GROUND_TRUTH_PATH}"
    )


    # ========================================================
    # Preview
    # ========================================================

    if not df.empty:

        print("\nตัวอย่างข้อมูล:")

        print(
            df.head(20)
        )


# ============================================================
# Natural Sort
# ============================================================

def natural_sort_key(filename):

    return [
        int(text)
        if text.isdigit()
        else text.lower()

        for text in re.split(
            r'(\d+)',
            filename
        )
    ]


# ============================================================
# Run Program
# ============================================================

if __name__ == "__main__":

    main()