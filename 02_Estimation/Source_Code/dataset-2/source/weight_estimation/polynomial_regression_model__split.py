# ======================================================
# Import Libraries
# ======================================================

import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures
)

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

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
    TEST_SIZE,
    VALIDATION_SIZE,
    DEBUG,
    POLYNOMIAL_REGRESSION_SPLIT_PATH
)


# ======================================================
# Main
# ======================================================

def main():

    # Create output folder
    POLYNOMIAL_REGRESSION_SPLIT_PATH.mkdir(
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
        print(f"Total Images : {len(model_df)}")

    # ======================================================
    # Prepare Features and Target
    # ======================================================

    # Input Features
    X = model_df[FEATURES]

    # Target
    y = model_df["Weight"]

    # Image names
    image_names = model_df["Image"]

    # ======================================================
    # Split Dataset
    # ======================================================

    # First split: separate Test Set
    X_train_valid, X_test, y_train_valid, y_test, image_train_valid, image_test = train_test_split(
        X,
        y,
        image_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Second split: separate Validation Set
    X_train, X_valid, y_train, y_valid, image_train, image_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        image_train_valid,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    if DEBUG:
        print(f"Train      : {len(X_train)}")
        print(f"Validation : {len(X_valid)}")
        print(f"Test       : {len(X_test)}")

    # ======================================================
    # Normalize Features
    # ======================================================

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_valid = scaler.transform(X_valid)

    X_test = scaler.transform(X_test)

    # ======================================================
    # Create Polynomial Features
    # ======================================================

    POLYNOMIAL_DEGREE = 2

    poly = PolynomialFeatures(
        degree=POLYNOMIAL_DEGREE,
        include_bias=False
    )

    X_train = poly.fit_transform(
        X_train
    )

    X_valid = poly.transform(
        X_valid
    )

    X_test = poly.transform(
        X_test
    )

    if DEBUG:
        print(f"Polynomial Degree : {POLYNOMIAL_DEGREE}")
        print(f"Original Features : {len(FEATURES)}")
        print(f"Polynomial Features : {X_train.shape[1]}")

    # ======================================================
    # Build Polynomial Regression Model
    # ======================================================

    model = LinearRegression()

    # ======================================================
    # Train Model
    # ======================================================

    model.fit(
        X_train,
        y_train
    )

    # ======================================================
    # Polynomial Feature Analysis
    # ======================================================

    polynomial_feature_names = (
        poly.get_feature_names_out(FEATURES)
    )

    coefficient_df = pd.DataFrame({
        "Feature": polynomial_feature_names,
        "Coefficient": model.coef_
    })

    coefficient_df = coefficient_df.sort_values(
        by="Coefficient",
        key=abs,
        ascending=False
    )

    if DEBUG:
        print(coefficient_df)

    # ======================================================
    # Feature Importance Plot
    # ======================================================

    plt.figure(
        figsize=(8, 5)
    )

    sns.barplot(
        data=coefficient_df.head(15),
        x="Coefficient",
        y="Feature"
    )

    plt.title(
        "Polynomial Regression Coefficient Analysis"
    )

    plt.xlabel("Coefficient")

    plt.ylabel("Feature")

    plt.grid(axis="x")

    plt.show()

    # ======================================================
    # Validation Prediction
    # ======================================================

    validation_predictions = model.predict(
        X_valid
    )

    # ======================================================
    # Validation Evaluation
    # ======================================================

    validation_mae = mean_absolute_error(
        y_valid,
        validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            validation_predictions
        )
    )

    validation_r2 = r2_score(
        y_valid,
        validation_predictions
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(
        f"Validation MAE  : {validation_mae:.4f}"
    )

    print(
        f"Validation RMSE : {validation_rmse:.4f}"
    )

    print(
        f"Validation R²   : {validation_r2:.4f}"
    )

    print("=" * 60)

    # ======================================================
    # Test Prediction
    # ======================================================

    predictions = model.predict(
        X_test
    )

    # ======================================================
    # Model Evaluation
    # ======================================================

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

    print()
    print("=" * 60)
    print("FINAL TEST RESULT")
    print("=" * 60)

    print(
        f"Test MAE  : {mae:.4f}"
    )

    print(
        f"Test RMSE : {rmse:.4f}"
    )

    print(
        f"Test R²   : {r2:.4f}"
    )

    print("=" * 60)

    # ======================================================
    # Prediction Results
    # ======================================================

    prediction_result_df = pd.DataFrame({
        "Image": image_test.values,
        "Actual": y_test.values,
        "Predicted": predictions
    })

    prediction_result_df["Error"] = (
        prediction_result_df["Predicted"]
        - prediction_result_df["Actual"]
    )

    prediction_result_df["Absolute Error"] = (
        prediction_result_df["Error"].abs()
    )

    if DEBUG:
        print(prediction_result_df)

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
    # Save Model, Scaler and Polynomial Transformer
    # ======================================================

    model_path = (
        POLYNOMIAL_REGRESSION_SPLIT_PATH
        / "polynomial_regression_split.joblib"
    )

    scaler_path = (
        POLYNOMIAL_REGRESSION_SPLIT_PATH
        / "polynomial_regression_split_scaler.joblib"
    )

    polynomial_path = (
        POLYNOMIAL_REGRESSION_SPLIT_PATH
        / "polynomial_regression_split_polynomial.joblib"
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
        poly,
        polynomial_path
    )

    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Polynomial Degree",
            "Original Features",
            "Polynomial Features",
            "Target",
            "Train Images",
            "Validation Images",
            "Test Images",
            "Random State",
            "Intercept"
        ],
        "Value": [
            "Polynomial Regression",
            POLYNOMIAL_DEGREE,
            len(FEATURES),
            X_train.shape[1],
            "Weight",
            len(y_train),
            len(y_valid),
            len(y_test),
            RANDOM_STATE,
            model.intercept_
        ]
    })

    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Feature": FEATURES,
        "Data Min": scaler.data_min_,
        "Data Max": scaler.data_max_,
        "Scale": scaler.scale_,
        "Min": scaler.min_
    })

    # ======================================================
    # Polynomial Transformer Information
    # ======================================================

    polynomial_info_df = pd.DataFrame({
        "Parameter": [
            "Polynomial Degree",
            "Original Features",
            "Polynomial Features",
            "Include Bias"
        ],
        "Value": [
            POLYNOMIAL_DEGREE,
            len(FEATURES),
            X_train.shape[1],
            False
        ]
    })

    # ======================================================
    # Save Results
    # ======================================================

    metrics_df = pd.DataFrame({
        "Dataset": [
            "Validation",
            "Test"
        ],
        "MAE": [
            validation_mae,
            mae
        ],
        "RMSE": [
            validation_rmse,
            rmse
        ],
        "R2": [
            validation_r2,
            r2
        ]
    })

    results_path = (
        POLYNOMIAL_REGRESSION_SPLIT_PATH
        / "polynomial_regression_split_results.xlsx"
    )

    with pd.ExcelWriter(results_path) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False
        )

        coefficient_df.to_excel(
            writer,
            sheet_name="Coefficients",
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
    # Actual vs Predicted Plot
    # ======================================================

    plt.figure(
        figsize=(6, 6)
    )

    sns.scatterplot(
        x=y_test,
        y=predictions,
        s=80,
        color="royalblue"
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
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
        "Polynomial Regression: Actual vs Predicted Weight"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Polynomial Regression - Split")
    print("=" * 60)

    print(
        f"Train                    : {len(X_train):,} ภาพ"
    )

    print(
        f"Validation               : {len(X_valid):,} ภาพ"
    )

    print(
        f"Test                     : {len(X_test):,} ภาพ"
    )

    print()

    print(
        f"Polynomial Degree        : {POLYNOMIAL_DEGREE}"
    )

    print(
        f"Polynomial Features      : {X_train.shape[1]:,}"
    )

    print()

    print(
        f"บันทึกโมเดลไว้ที่         : {model_path}"
    )

    print(
        f"บันทึก Scaler ไว้ที่      : {scaler_path}"
    )

    print(
        f"บันทึก Polynomial ไว้ที่  : {polynomial_path}"
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