# ======================================================
# Import Libraries
# ======================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
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
    DEBUG,
    POLYNOMIAL_REGRESSION_KFOLD_PATH
)


# ======================================================
# Main
# ======================================================

def main():

    # Create output folder
    POLYNOMIAL_REGRESSION_KFOLD_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create fold model folder
    fold_models_path = (
        POLYNOMIAL_REGRESSION_KFOLD_PATH
        / "fold_models"
    )

    fold_models_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read dataset
    model_df = pd.read_excel(
        MODEL_DATASET_PATH
    )


    # ======================================================
    # Dataset Information
    # ======================================================

    if DEBUG:
        print("=" * 60)
        print("POLYNOMIAL REGRESSION - K-FOLD")
        print("=" * 60)
        print(
            f"Total Images : {len(model_df):,}"
        )
        print(
            f"Total Pigs   : "
            f"{model_df['Pig_ID'].nunique():,}"
        )
        print(
            f"K-Fold       : {KFOLD_SPLITS}"
        )
        print(
            "Polynomial Degree : 2"
        )


    # ======================================================
    # Prepare Features and Target
    # ======================================================

    # Input Features
    X = model_df[FEATURES]

    # Target
    y = model_df["Weight"]

    # Pig ID for Group K-Fold
    groups = model_df["Pig_ID"]

    # Image names
    image_names = model_df["Image"]


    # ======================================================
    # K-Fold Split
    # ======================================================

    group_kfold = GroupKFold(
        n_splits=KFOLD_SPLITS
    )

    fold_results = []
    prediction_results = []

    fold_mae = []
    fold_rmse = []


    # ======================================================
    # Train and Test Each Fold
    # ======================================================

    for fold_number, (train_index, test_index) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        print()
        print("=" * 60)
        print(
            f"FOLD {fold_number:02d}/{KFOLD_SPLITS}"
        )
        print("=" * 60)


        # ==================================================
        # Prepare Train/Test Data
        # ==================================================

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        image_test = image_names.iloc[test_index]
        pig_test = groups.iloc[test_index]


        if DEBUG:
            print(
                f"Train Images : {len(X_train):,}"
            )
            print(
                f"Test Images  : {len(X_test):,}"
            )
            print(
                f"Train Pigs   : "
                f"{groups.iloc[train_index].nunique():,}"
            )
            print(
                f"Test Pigs    : "
                f"{pig_test.nunique():,}"
            )


        # ==================================================
        # Normalize Features
        # ==================================================

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ==================================================
        # Polynomial Transformation
        # ==================================================

        polynomial = PolynomialFeatures(
            degree=2,
            include_bias=False
        )

        X_train_poly = polynomial.fit_transform(
            X_train_scaled
        )

        X_test_poly = polynomial.transform(
            X_test_scaled
        )


        # ==================================================
        # Build Polynomial Regression Model
        # ==================================================

        model = LinearRegression()


        # ==================================================
        # Train Model
        # ==================================================

        model.fit(
            X_train_poly,
            y_train
        )


        # ==================================================
        # Test Prediction
        # ==================================================

        predictions = model.predict(
            X_test_poly
        )


        # ==================================================
        # Model Evaluation
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

        fold_mae.append(
            mae
        )

        fold_rmse.append(
            rmse
        )


        print(
            f"Test MAE  : {mae:.4f}"
        )
        print(
            f"Test RMSE : {rmse:.4f}"
        )
        print(
            f"Test R²   : {r2:.4f}"
        )


        # ==================================================
        # Fold Results
        # ==================================================

        fold_results.append({
            "Fold": fold_number,
            "Train Images": len(X_train),
            "Test Images": len(X_test),
            "Train Pigs": groups.iloc[
                train_index
            ].nunique(),
            "Test Pigs": pig_test.nunique(),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


        # ==================================================
        # Prediction Results
        # ==================================================

        fold_prediction_df = pd.DataFrame({
            "Fold": fold_number,
            "Image": image_test.values,
            "Pig_ID": pig_test.values,
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


        # ==================================================
        # Save Fold Model, Scaler and Polynomial
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
            fold_path
            / "model.joblib"
        )

        scaler_path = (
            fold_path
            / "scaler.joblib"
        )

        polynomial_path = (
            fold_path
            / "polynomial.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        joblib.dump(
            polynomial,
            polynomial_path
        )


    # ======================================================
    # Combine Fold Results
    # ======================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    prediction_result_df = pd.concat(
        prediction_results,
        ignore_index=True
    )


    # ======================================================
    # Overall Metrics
    # ======================================================

    average_mae = np.mean(
        fold_mae
    )

    average_rmse = np.mean(
        fold_rmse
    )

    overall_mae = mean_absolute_error(
        prediction_result_df["Actual"],
        prediction_result_df["Predicted"]
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            prediction_result_df["Actual"],
            prediction_result_df["Predicted"]
        )
    )

    overall_r2 = r2_score(
        prediction_result_df["Actual"],
        prediction_result_df["Predicted"]
    )


    metrics_df = pd.DataFrame({
        "Metric": [
            "Number of Images",
            "Number of Pigs",
            "K-Fold Splits",
            "Polynomial Degree",
            "Average MAE",
            "Average RMSE",
            "Overall MAE",
            "Overall RMSE",
            "Overall R2"
        ],

        "Value": [
            len(model_df),
            model_df["Pig_ID"].nunique(),
            KFOLD_SPLITS,
            2,
            average_mae,
            average_rmse,
            overall_mae,
            overall_rmse,
            overall_r2
        ]
    })


    # ======================================================
    # Model Information
    # ======================================================

    polynomial_feature_count = (
        len(FEATURES)
        + (
            len(FEATURES)
            * (len(FEATURES) + 1)
            // 2
        )
    )

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Number of Features",
            "Polynomial Degree",
            "Polynomial Features",
            "Target",
            "K-Fold Method",
            "Group",
            "K-Fold Splits",
            "Random State"
        ],

        "Value": [
            "Polynomial Regression",
            len(FEATURES),
            2,
            polynomial_feature_count,
            "Weight",
            "GroupKFold",
            "Pig_ID",
            KFOLD_SPLITS,
            RANDOM_STATE
        ]
    })


    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Feature": FEATURES
    })


    # ======================================================
    # Polynomial Information
    # ======================================================

    polynomial_info_df = pd.DataFrame({
        "Parameter": [
            "Polynomial Degree",
            "Include Bias",
            "Original Features",
            "Generated Features"
        ],

        "Value": [
            2,
            False,
            len(FEATURES),
            polynomial_feature_count
        ]
    })


    # ======================================================
    # Round Values
    # ======================================================

    prediction_result_df["Predicted"] = (
        prediction_result_df["Predicted"]
        .round(2)
    )

    prediction_result_df["Error"] = (
        prediction_result_df["Error"]
        .round(2)
    )

    prediction_result_df["Absolute Error"] = (
        prediction_result_df["Absolute Error"]
        .round(2)
    )


    # ======================================================
    # Save Excel Results
    # ======================================================

    results_path = (
        POLYNOMIAL_REGRESSION_KFOLD_PATH
        / "polynomial_regression_kfold_results.xlsx"
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

        prediction_result_df.to_excel(
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

        polynomial_info_df.to_excel(
            writer,
            sheet_name="Polynomial_Info",
            index=False
        )


    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Polynomial Regression - K-Fold")
    print("=" * 60)

    print(
        f"Number of Images        : "
        f"{len(model_df):,} ภาพ"
    )

    print(
        f"Number of Pigs          : "
        f"{model_df['Pig_ID'].nunique():,} ตัว"
    )

    print(
        f"K-Fold                  : "
        f"{KFOLD_SPLITS} Folds"
    )

    print(
        f"Polynomial Degree       : 2"
    )

    print()
    print(
        f"Average MAE             : "
        f"{average_mae:.4f} kg"
    )

    print(
        f"Average RMSE            : "
        f"{average_rmse:.4f} kg"
    )

    print(
        f"Overall MAE             : "
        f"{overall_mae:.4f} kg"
    )

    print(
        f"Overall RMSE            : "
        f"{overall_rmse:.4f} kg"
    )

    print(
        f"Overall R²              : "
        f"{overall_r2:.4f}"
    )

    print()
    print(
        f"บันทึก Fold Models ไว้ที่ : "
        f"{fold_models_path}"
    )

    print(
        f"บันทึกผลลัพธ์ไว้ที่       : "
        f"{results_path}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
