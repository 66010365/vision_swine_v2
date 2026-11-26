# ======================================================
# Import Libraries
# ======================================================

import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
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
    LINEAR_REGRESSION_SPLIT_PATH
)

# ======================================================
# Main
# ======================================================

def main():

    # Create output folder
    LINEAR_REGRESSION_SPLIT_PATH.mkdir(
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
    # Build Linear Regression Model
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
    # Train Prediction
    # ======================================================

    train_predictions = model.predict(
        X_train
    )

    # ======================================================
    # Train Evaluation
    # ======================================================

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_predictions
        )
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    print()
    print("=" * 60)
    print("TRAIN RESULT")
    print("=" * 60)
    print(f"Train MAE  : {train_mae:.4f}")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Train R²   : {train_r2:.4f}")
    print("=" * 60)

    # ======================================================
    # Feature Importance Analysis
    # ======================================================
    
    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
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
        data=coefficient_df,
        x="Coefficient",
        y="Feature"
    )

    plt.title(
        "Linear Regression Feature Importance"
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
    print(f"Validation MAE  : {validation_mae:.4f}")
    print(f"Validation RMSE : {validation_rmse:.4f}")
    print(f"Validation R²   : {validation_r2:.4f}")
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
    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test R²   : {r2:.4f}")
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
    # Save Model and Scaler
    # ======================================================

    model_path = (
        LINEAR_REGRESSION_SPLIT_PATH
        / "linear_regression_split.joblib"
    )

    scaler_path = (
        LINEAR_REGRESSION_SPLIT_PATH
        / "linear_regression_split_scaler.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )


    # ======================================================
    # Save Results
    # ======================================================

    metrics_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "MAE": [train_mae, validation_mae, mae],
        "RMSE": [train_rmse, validation_rmse, rmse],
        "R2": [train_r2, validation_r2, r2]
    })


    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Number of Features",
            "Target",
            "Train Images",
            "Validation Images",
            "Test Images",
            "Random State",
            "Train MAE",
            "Train RMSE",
            "Train R2",
            "Validation MAE",
            "Validation RMSE",
            "Validation R2",
            "Test MAE",
            "Test RMSE",
            "Test R2",
            "Intercept"
        ],
        "Value": [
            "Linear Regression",
            len(FEATURES),
            "Weight",
            len(X_train),
            len(X_valid),
            len(X_test),
            RANDOM_STATE,
            train_mae,
            train_rmse,
            train_r2,
            validation_mae,
            validation_rmse,
            validation_r2,
            mae,
            rmse,
            r2,
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
    # Save Excel Results
    # ======================================================

    results_path = (
        LINEAR_REGRESSION_SPLIT_PATH
        / "linear_regression_split_results.xlsx"
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

    plt.xlabel("Actual Weight (kg)")
    plt.ylabel("Predicted Weight (kg)")
    plt.title("Actual vs Predicted Weight")

    plt.legend()
    plt.grid(True)

    plt.show()

    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Linear Regression - Split")
    print("=" * 60)
    print(f"Train                    : {len(X_train):,} ภาพ")
    print(f"Validation               : {len(X_valid):,} ภาพ")
    print(f"Test                     : {len(X_test):,} ภาพ")
    print()
    print("Overfitting Check")
    print(f"Train → Validation ΔR²  : {train_r2 - validation_r2:.4f}")
    print(f"Train → Test ΔR²        : {train_r2 - r2:.4f}")
    print(f"Validation → Test ΔR²   : {validation_r2 - r2:.4f}")
    print()
    print(f"บันทึกโมเดลไว้ที่         : {model_path}")
    print(f"บันทึก Scaler ไว้ที่      : {scaler_path}")
    print(f"บันทึกผลลัพธ์ไว้ที่       : {results_path}")
    print("=" * 60)

# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
