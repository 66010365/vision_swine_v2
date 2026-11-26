# ======================================================
# Import Libraries
# ======================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut

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
    DEBUG,
    POLYNOMIAL_REGRESSION_LOGO_PATH
)


# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Create Output Folder
    # ======================================================

    POLYNOMIAL_REGRESSION_LOGO_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Folder for saving model artifacts of every LOGO fold
    fold_models_path = POLYNOMIAL_REGRESSION_LOGO_PATH / "fold_models"
    fold_models_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # Read Dataset
    # ======================================================

    model_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    # ======================================================
    # Dataset Information
    # ======================================================

    if DEBUG:
        print(
            f"Total Images : {len(model_df)}"
        )

        print("\nImages per Pig")

        print(
            model_df["Pig_ID"].value_counts()
        )

    # ======================================================
    # Input Features
    # ======================================================

    X = model_df[FEATURES]

    # Target
    y = model_df["Weight"]

    # Groups
    groups = model_df["Pig_ID"]

    # Image names
    image_names = model_df["Image"]

    # ======================================================
    # Polynomial Degree
    # ======================================================

    POLYNOMIAL_DEGREE = 2

    # ======================================================
    # Leave-One-Group-Out (LOGO)
    # ======================================================

    logo = LeaveOneGroupOut()

    mae_scores = []
    rmse_scores = []

    prediction_results = []
    logo_results = []

    # Store model, scaler and polynomial information for every LOGO fold
    model_info_results = []
    scaler_info_results = []
    polynomial_info_results = []

    # ======================================================
    # LOGO Training
    # ======================================================

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        # ==================================================
        # Separate Train and Test
        # ==================================================

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        image_test = image_names.iloc[test_idx]

        test_pig = groups.iloc[test_idx].iloc[0]

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
        # Create Polynomial Features
        # ==================================================

        poly = PolynomialFeatures(
            degree=POLYNOMIAL_DEGREE,
            include_bias=False
        )

        X_train = poly.fit_transform(
            X_train
        )

        X_test = poly.transform(
            X_test
        )

        # ==================================================
        # Build Polynomial Regression Model
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
        # Save Model, Scaler and Polynomial Transformer
        # ==================================================

        fold_path = fold_models_path / f"fold_{fold:02d}"
        fold_path.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = fold_path / "model.joblib"
        scaler_path = fold_path / "scaler.joblib"
        polynomial_path = fold_path / "polynomial.joblib"

        joblib.dump(
            model,
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        joblib.dump(
            poly,
            polynomial_path
        )

        # ==================================================
        # Store Model Information
        # ==================================================

        model_info_results.append({
            "Fold": fold,
            "Test_Pig_ID": test_pig,
            "Number of Training Images": len(train_idx),
            "Number of Test Images": len(test_idx),
            "Model Path": str(model_path),
            "Scaler Path": str(scaler_path),
            "Polynomial Path": str(polynomial_path),
            "Intercept": model.intercept_,
            **{
                f"Coefficient_{feature}": coefficient
                for feature, coefficient in zip(
                    poly.get_feature_names_out(FEATURES),
                    model.coef_
                )
            }
        })

        # ==================================================
        # Store Scaler Information
        # ==================================================

        for feature, data_min, data_max, scale, min_value in zip(
            FEATURES,
            scaler.data_min_,
            scaler.data_max_,
            scaler.scale_,
            scaler.min_
        ):
            scaler_info_results.append({
                "Fold": fold,
                "Test_Pig_ID": test_pig,
                "Feature": feature,
                "Data Min": data_min,
                "Data Max": data_max,
                "Scale": scale,
                "Min": min_value
            })

        # ==================================================
        # Store Polynomial Information
        # ==================================================

        polynomial_info_results.append({
            "Fold": fold,
            "Test_Pig_ID": test_pig,
            "Polynomial Degree": POLYNOMIAL_DEGREE,
            "Original Features": len(FEATURES),
            "Polynomial Features": X_train.shape[1],
            "Include Bias": False
        })

        # ==================================================
        # Prediction
        # ==================================================

        predictions = model.predict(
            X_test
        )

        # ==================================================
        # Evaluate Model
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

        # ==================================================
        # Store Scores
        # ==================================================

        mae_scores.append(
            mae
        )

        rmse_scores.append(
            rmse
        )

        logo_results.append({
            "Fold": fold,
            "Pig_ID": test_pig,
            "Images": len(test_idx),
            "MAE": mae,
            "RMSE": rmse
        })

        # ==================================================
        # Prediction Results
        # ==================================================

        prediction_df = pd.DataFrame({
            "Fold": fold,
            "Image": image_test.values,
            "Pig_ID": groups.iloc[test_idx].values,
            "Actual": y_test.values,
            "Predicted": predictions
        })

        prediction_df["Error"] = (
            prediction_df["Predicted"]
            - prediction_df["Actual"]
        )

        prediction_df["Absolute Error"] = (
            prediction_df["Error"].abs()
        )

        prediction_results.append(
            prediction_df
        )

        # ==================================================
        # DEBUG
        # ==================================================

        if DEBUG:
            print(
                f"Fold {fold:02d}/{groups.nunique():02d} | "
                f"Pig_ID: {test_pig} | "
                f"Images: {len(test_idx)} | "
                f"MAE: {mae:.4f} | "
                f"RMSE: {rmse:.4f}"
            )

    # ======================================================
    # LOGO Results per Pig
    # ======================================================

    logo_result_df = pd.DataFrame(
        logo_results
    )

    print()
    print("=" * 60)
    print("LOGO RESULTS PER PIG")
    print("=" * 60)

    print(
        logo_result_df
    )

    # ======================================================
    # LOGO Evaluation Summary
    # ======================================================

    mean_mae = np.mean(
        mae_scores
    )

    mean_rmse = np.mean(
        rmse_scores
    )

    print()
    print("=" * 60)
    print("LOGO EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Polynomial Degree : {POLYNOMIAL_DEGREE}"
    )

    print(
        f"Number of Pigs    : {groups.nunique()}"
    )

    print(
        f"Average MAE       : {mean_mae:.4f}"
    )

    print(
        f"Average RMSE      : {mean_rmse:.4f}"
    )

    # ======================================================
    # Combine Prediction Results
    # ======================================================

    prediction_df = pd.concat(
        prediction_results,
        ignore_index=True
    )

    # ======================================================
    # Overall LOGO Evaluation
    # ======================================================

    overall_mae = mean_absolute_error(
        prediction_df["Actual"],
        prediction_df["Predicted"]
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            prediction_df["Actual"],
            prediction_df["Predicted"]
        )
    )

    overall_r2 = r2_score(
        prediction_df["Actual"],
        prediction_df["Predicted"]
    )

    print()
    print("=" * 60)
    print("OVERALL LOGO EVALUATION")
    print("=" * 60)

    print(
        f"Overall MAE  : {overall_mae:.4f}"
    )

    print(
        f"Overall RMSE : {overall_rmse:.4f}"
    )

    print(
        f"Overall R²   : {overall_r2:.4f}"
    )

    print("=" * 60)

    # ======================================================
    # Round Prediction Results
    # ======================================================

    prediction_df["Predicted"] = (
        prediction_df["Predicted"]
        .round(2)
    )

    prediction_df["Error"] = (
        prediction_df["Error"]
        .round(2)
    )

    prediction_df["Absolute Error"] = (
        prediction_df["Absolute Error"]
        .round(2)
    )

    # ======================================================
    # Print Prediction Results
    # ======================================================

    if DEBUG:
        print(
            prediction_df
        )

    # ======================================================
    # Metrics
    # ======================================================

    metrics_df = pd.DataFrame({
        "Metric": [
            "Polynomial Degree",
            "Number of Pigs",
            "Total Images",
            "Average MAE",
            "Average RMSE",
            "Overall MAE",
            "Overall RMSE",
            "Overall R2"
        ],
        "Value": [
            POLYNOMIAL_DEGREE,
            groups.nunique(),
            len(model_df),
            mean_mae,
            mean_rmse,
            overall_mae,
            overall_rmse,
            overall_r2
        ]
    })

    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame(
        model_info_results
    )

    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame(
        scaler_info_results
    )

    # ======================================================
    # Polynomial Transformer Information
    # ======================================================

    polynomial_info_df = pd.DataFrame(
        polynomial_info_results
    )

    # ======================================================
    # Save Results
    # ======================================================

    results_path = (
        POLYNOMIAL_REGRESSION_LOGO_PATH
        / "polynomial_regression_logo_results.xlsx"
    )

    with pd.ExcelWriter(
        results_path
    ) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False
        )

        logo_result_df.to_excel(
            writer,
            sheet_name="LOGO_Per_Pig",
            index=False
        )

        prediction_df.to_excel(
            writer,
            sheet_name="Predictions",
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
    # Actual vs Predicted
    # ======================================================

    plt.figure(
        figsize=(6, 6)
    )

    sns.scatterplot(
        data=prediction_df,
        x="Actual",
        y="Predicted",
        s=80,
        color="royalblue"
    )

    plt.plot(
        [
            prediction_df["Actual"].min(),
            prediction_df["Actual"].max()
        ],
        [
            prediction_df["Actual"].min(),
            prediction_df["Actual"].max()
        ],
        "r--",
        linewidth=2,
        label="Ideal Prediction"
    )

    plt.xlabel(
        "Actual Weight (kg)"
    )

    plt.ylabel(
        "Predicted Weight (kg)"
    )

    plt.title(
        "Polynomial Regression - LOGO"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Polynomial Regression - LOGO")
    print("=" * 60)

    print(
        f"จำนวนสุกรทั้งหมด       : {groups.nunique():,} ตัว"
    )

    print(
        f"จำนวนภาพทั้งหมด        : {len(model_df):,} ภาพ"
    )

    print(
        f"Polynomial Degree      : {POLYNOMIAL_DEGREE}"
    )

    print()

    print(
        f"Average MAE            : {mean_mae:.4f}"
    )

    print(
        f"Average RMSE           : {mean_rmse:.4f}"
    )

    print()

    print(
        f"Overall MAE            : {overall_mae:.4f}"
    )

    print(
        f"Overall RMSE           : {overall_rmse:.4f}"
    )

    print(
        f"Overall R²             : {overall_r2:.4f}"
    )

    print()

    print(
        f"บันทึก Model/Scaler/Polynomial ไว้ที่ : {fold_models_path}"
    )

    print(
        f"บันทึกผลลัพธ์ไว้ที่     : {results_path}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()