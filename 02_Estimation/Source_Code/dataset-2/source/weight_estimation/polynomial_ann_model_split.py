# ======================================================
# Import Libraries
# ======================================================

import random
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ======================================================
# Import Configuration
# ======================================================

from config import (
    MODEL_DATASET_PATH,
    FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    DEBUG,
    POLYNOMIAL_ANN_SPLIT_PATH
)


# ======================================================
# Set Random Seed
# ======================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ======================================================
# Main
# ======================================================

def main():

    # Create output folder
    POLYNOMIAL_ANN_SPLIT_PATH.mkdir(
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
        print(f"Polynomial Degree   : {POLYNOMIAL_DEGREE}")
        print(f"Original Features   : {len(FEATURES)}")
        print(f"Polynomial Features : {X_train.shape[1]}")

    # ======================================================
    # Build Polynomial ANN Model
    # ======================================================

    model = Sequential([
        Dense(
            HIDDEN_LAYER_1,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),
        Dense(
            HIDDEN_LAYER_2,
            activation="relu"
        ),
        Dense(
            1,
            activation="linear"
        )
    ])

    optimizer = Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    # ======================================================
    # Train Model
    # ======================================================

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2
    )

    # ======================================================
    # Training History
    # ======================================================

    best_epoch = np.argmin(
        history.history["val_loss"]
    ) + 1

    training_epochs = len(
        history.history["loss"]
    )

    # ======================================================
    # Validation Prediction
    # ======================================================

    validation_predictions = model.predict(
        X_valid,
        verbose=0
    ).flatten()

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
        X_test,
        verbose=0
    ).flatten()

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
        POLYNOMIAL_ANN_SPLIT_PATH
        / "polynomial_ann_split.keras"
    )

    scaler_path = (
        POLYNOMIAL_ANN_SPLIT_PATH
        / "polynomial_ann_split_scaler.joblib"
    )

    polynomial_path = (
        POLYNOMIAL_ANN_SPLIT_PATH
        / "polynomial_ann_split_polynomial.joblib"
    )

    model.save(
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
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output Layer",
            "Learning Rate",
            "Batch Size",
            "Maximum Epochs",
            "Training Epochs",
            "Best Epoch",
            "Target",
            "Train Images",
            "Validation Images",
            "Test Images",
            "Random State"
        ],
        "Value": [
            "Polynomial ANN",
            POLYNOMIAL_DEGREE,
            len(FEATURES),
            X_train.shape[1],
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            1,
            LEARNING_RATE,
            BATCH_SIZE,
            EPOCHS,
            training_epochs,
            best_epoch,
            "Weight",
            len(y_train),
            len(y_valid),
            len(y_test),
            RANDOM_STATE
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
    # Training History Information
    # ======================================================

    history_df = pd.DataFrame({
        "Epoch": np.arange(
            1,
            training_epochs + 1
        ),
        "Loss": history.history["loss"],
        "MAE": history.history["mae"],
        "Validation Loss": history.history["val_loss"],
        "Validation MAE": history.history["val_mae"]
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
        POLYNOMIAL_ANN_SPLIT_PATH
        / "polynomial_ann_split_results.xlsx"
    )

    with pd.ExcelWriter(results_path) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False
        )

        history_df.to_excel(
            writer,
            sheet_name="Training_History",
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
    # Training History Plot
    # ======================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("MSE Loss")

    plt.title(
        "Polynomial ANN Training History"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # ======================================================
    # Actual vs Predicted Plot
    # ======================================================

    plt.figure(
        figsize=(6, 6)
    )

    sns.scatterplot(
        x=y_test,
        y=predictions,
        s=80
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
        "Polynomial ANN: Actual vs Predicted Weight"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Polynomial ANN - Split")
    print("=" * 60)

    print(
        f"Train                    : {len(X_train_valid) - len(X_valid):,} ภาพ"
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

    print(
        f"Training Epochs          : {training_epochs:,}"
    )

    print(
        f"Best Epoch              : {best_epoch:,}"
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
