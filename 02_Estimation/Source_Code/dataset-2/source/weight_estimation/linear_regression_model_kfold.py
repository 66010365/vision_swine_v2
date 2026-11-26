# ======================================================
# Import Libraries
# ======================================================

import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ======================================================
# Import Configuration
# ======================================================

from config import (
    MODEL_DATASET_PATH,
    FEATURES,
    RANDOM_STATE,
    KFOLD_SPLITS,
    LINEAR_REGRESSION_KFOLD_PATH,
    DEBUG
)


# ======================================================
# Random Seed
# ======================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Create Output Folder
    # ======================================================

    output_path = LINEAR_REGRESSION_KFOLD_PATH

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_models_path = (
        output_path / "fold_models"
    )

    fold_models_path.mkdir(
        parents=True,
        exist_ok=True
    )


    # ======================================================
    # Read Regression Dataset
    # ======================================================

    regression_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    required_columns = FEATURES + [
        "Pig_ID",
        "Weight"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in regression_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    regression_df = regression_df.dropna(
        subset=required_columns
    ).reset_index(drop=True)


    # ======================================================
    # Input Features
    # ======================================================

    X = regression_df[FEATURES]


    # ======================================================
    # Target
    # ======================================================

    y = regression_df["Weight"]


    # ======================================================
    # Image Names
    # ======================================================

    image_names = regression_df["Image"]


    # ======================================================
    # Pig Groups
    # ======================================================

    groups = regression_df["Pig_ID"]

    pig_ids = sorted(
        groups.unique()
    )


    print()
    print("=" * 60)
    print("LINEAR REGRESSION - GROUP K-FOLD")
    print("=" * 60)
    print()

    print("Dataset")
    print("-" * 60)
    print(f"Total Images       : {len(regression_df):,}")
    print(f"Total Pigs         : {len(pig_ids)}")
    print(f"Number of Folds    : {KFOLD_SPLITS}")


    # ======================================================
    # K-Fold Configuration
    # ======================================================

    group_kfold = GroupKFold(
        n_splits=KFOLD_SPLITS
    )


    # ======================================================
    # Results
    # ======================================================

    fold_results = []
    prediction_results = []


    # ======================================================
    # K-Fold Training
    # ======================================================

    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        print()
        print("=" * 60)
        print(
            f"Fold {fold_number:02d}/{KFOLD_SPLITS}"
        )
        print("=" * 60)


        # ==================================================
        # Train / Test Dataset
        # ==================================================

        X_train = X.iloc[train_index].copy()
        y_train = y.iloc[train_index].copy()

        X_test = X.iloc[test_index].copy()
        y_test = y.iloc[test_index].copy()

        image_test = image_names.iloc[test_index].copy()

        groups_test = groups.iloc[test_index].copy()


        # ==================================================
        # Normalize Features
        # ==================================================

        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )


        # ==================================================
        # Build Linear Regression Model
        # ==================================================

        model = LinearRegression()


        # ==================================================
        # Train Model
        # ==================================================

        model.fit(
            X_train,
            y_train
        )


        # ==================================================
        # Predict Weight
        # ==================================================

        predictions = model.predict(
            X_test
        )


        # ==================================================
        # Calculate Metrics
        # ==================================================

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        print(
            f"Test MAE  : {mae:.4f} kg"
        )

        print(
            f"Test RMSE : {rmse:.4f} kg"
        )

        print(
            f"Test R²   : {r2:.4f}"
        )


        # ==================================================
        # Save Fold Model / Scaler
        # ==================================================

        fold_path = (
            fold_models_path
            / f"fold_{fold_number:02d}"
        )

        fold_path.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            fold_path / "model.joblib"
        )

        scaler_path = (
            fold_path / "scaler.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )


        # ==================================================
        # Fold Results
        # ==================================================

        fold_results.append({
            "Fold": fold_number,
            "Test Pigs": ", ".join(
                sorted(
                    groups_test.unique()
                )
            ),
            "Train Images": len(train_index),
            "Test Images": len(test_index),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Model Path": str(model_path),
            "Scaler Path": str(scaler_path)
        })


        # ==================================================
        # Prediction Results
        # ==================================================

        fold_prediction_df = pd.DataFrame({
            "Fold": fold_number,
            "Pig_ID": groups_test.values,
            "Image": image_test.values,
            "Actual": y_test.values,
            "Predicted": predictions
        })

        fold_prediction_df["Error"] = (
            fold_prediction_df["Predicted"]
            - fold_prediction_df["Actual"]
        )

        fold_prediction_df["Absolute Error"] = (
            fold_prediction_df["Error"].abs()
        )

        prediction_results.append(
            fold_prediction_df
        )


    # ======================================================
    # Combine Results
    # ======================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    prediction_results_df = pd.concat(
        prediction_results,
        ignore_index=True
    )


    # ======================================================
    # Overall Metrics
    # ======================================================

    average_mae = (
        fold_results_df["MAE"].mean()
    )

    average_rmse = (
        fold_results_df["RMSE"].mean()
    )

    overall_mae = mean_absolute_error(
        prediction_results_df["Actual"],
        prediction_results_df["Predicted"]
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            prediction_results_df["Actual"],
            prediction_results_df["Predicted"]
        )
    )

    overall_r2 = r2_score(
        prediction_results_df["Actual"],
        prediction_results_df["Predicted"]
    )


    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Validation Method",
            "Number of Folds",
            "Input Features",
            "Target",
            "Scaler",
            "Random State",
            "Fold Models Path"
        ],
        "Value": [
            "Linear Regression",
            "Group K-Fold",
            KFOLD_SPLITS,
            ", ".join(FEATURES),
            "Weight",
            "MinMaxScaler",
            RANDOM_STATE,
            str(fold_models_path)
        ]
    })


    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Parameter": [
            "Scaler",
            "Scaler Fit",
            "Scaler File"
        ],
        "Value": [
            "MinMaxScaler",
            "Training Data Only",
            "One scaler per K-Fold"
        ]
    })


    # ======================================================
    # Metrics Information
    # ======================================================

    metrics_df = pd.DataFrame({
        "Metric": [
            "Number of Pigs",
            "Number of Images",
            "Number of Folds",
            "Average MAE (kg)",
            "Average RMSE (kg)",
            "Overall MAE (kg)",
            "Overall RMSE (kg)",
            "Overall R²"
        ],
        "Value": [
            len(pig_ids),
            len(regression_df),
            KFOLD_SPLITS,
            average_mae,
            average_rmse,
            overall_mae,
            overall_rmse,
            overall_r2
        ]
    })


    # ======================================================
    # Save Results
    # ======================================================

    results_path = (
        output_path
        / "linear_regression_kfold_results.xlsx"
    )

    with pd.ExcelWriter(
        results_path
    ) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False
        )

        fold_results_df.to_excel(
            writer,
            sheet_name="Fold_Results",
            index=False
        )

        prediction_results_df.to_excel(
            writer,
            sheet_name="Test_Predictions",
            index=False
        )

        model_info_df.to_excel(
            writer,
            sheet_name="Model_Info",
            index=False
        )

        scaler_info_df.to_excel(
            writer,
            sheet_name="Scaler_Info",
            index=False
        )


    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("LINEAR REGRESSION - GROUP K-FOLD RESULT")
    print("=" * 60)

    print(
        f"Number of Pigs     : {len(pig_ids)}"
    )

    print(
        f"Number of Images   : {len(regression_df):,}"
    )

    print(
        f"Number of Folds    : {KFOLD_SPLITS}"
    )

    print(
        f"Average MAE        : {average_mae:.4f} kg"
    )

    print(
        f"Average RMSE       : {average_rmse:.4f} kg"
    )

    print(
        f"Overall MAE        : {overall_mae:.4f} kg"
    )

    print(
        f"Overall RMSE       : {overall_rmse:.4f} kg"
    )

    print(
        f"Overall R²         : {overall_r2:.4f}"
    )

    print()

    print(
        f"บันทึก Fold Models ไว้ที่ : {fold_models_path}"
    )

    print(
        f"บันทึกผลลัพธ์ไว้ที่       : {results_path}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
