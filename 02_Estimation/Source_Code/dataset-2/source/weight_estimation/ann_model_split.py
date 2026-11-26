# ======================================================
# Import Libraries
# ======================================================

import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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
    ANN_SPLIT_PATH
)

# ======================================================
# Random Seed
# ======================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Create Output Folder
    # ======================================================

    ANN_SPLIT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read ANN dataset
    ann_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    # ======================================================
    # Exploratory Data Analysis (EDA)
    # ======================================================

    # Correlation Analysis
    correlation = ann_df.corr(
        numeric_only=True
    )

    # ======================================================
    # Correlation Heatmap
    # ======================================================
    # Heatmap
    plt.figure(
        figsize=(10, 8)
        
    )
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    
    plt.title(
        "Feature Correlation Matrix"
    )
    
    plt.show()

    # ======================================================
    # Scatter Plot
    # ======================================================
  
    # Features for EDA
    feature_names = [
        "Area",
        "Perimeter",
        "Bounding_Box_Length",
        "Bounding_Box_Width",
        "Deviation",
        "Skeleton_Length",
        "Straight_Distance",
        "Straightness"
    ]

    for feature in feature_names:

        plt.figure(
            figsize=(6, 4)
        )

        sns.regplot(
            data=ann_df,
            x=feature,
            y="Weight",
            scatter_kws={
                "s": 60
            },
            line_kws={
            "color": "red"
            }
    )

        plt.title(f"{feature} vs Weight")

        plt.xlabel(feature)
        plt.ylabel("Weight")

        plt.grid(True)

        plt.show()

    # ======================================================
    # Input Features
    # ======================================================

    X = ann_df[FEATURES]

    # Target
    y = ann_df["Weight"]


    # Image names
    image_names = ann_df["Image"]

    # ======================================================
    # Split Dataset
    # ======================================================
    
    X_train, X_test, y_train, y_test, image_train, image_test = train_test_split(
        X,
        y,
        image_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ======================================================
    # Split Train / Validation
    # ======================================================

    X_train, X_val, y_train, y_val, image_train, image_val = train_test_split(
        X_train,
        y_train,
        image_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    print()
    print("=" * 60)
    print("ANN REGRESSION - SPLIT")
    print("=" * 60)
    print()
    print("Dataset")
    print("-" * 60)
    print(f"Total Images       : {len(ann_df):,}")
    print(f"Train              : {len(X_train):,}")
    print(f"Validation         : {len(X_val):,}")
    print(f"Test               : {len(X_test):,}")

    # ======================================================
    # Normalize Features
    # ======================================================
    
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)



    # ======================================================
    # Build ANN Model
    # ======================================================
    
    model = Sequential()

    model.add(
        Input(shape=(len(FEATURES),))
    )

    model.add(
        Dense(
            HIDDEN_LAYER_1,
            activation="relu"
        )
    )

    model.add(
        Dense(
            HIDDEN_LAYER_2,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1
        )  
    )
    print()
    print("Model Architecture")
    print("-" * 60)
    print(f"Input Features     : {len(FEATURES)}")
    print(f"Hidden Layer 1     : {HIDDEN_LAYER_1} neurons")
    print(f"Hidden Layer 2     : {HIDDEN_LAYER_2} neurons")
    print("Output             : 1 neuron")
    print(f"Total Parameters   : {model.count_params():,}")

    # ======================================================
    # Compile ANN Model
    # ======================================================
    
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=["mae"]
    )

    # ======================================================
    # Early Stopping
    # ======================================================
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )  
    
    # ======================================================
    # Train ANN Model
    # ======================================================
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=2
    )

    # ======================================================
    # Training Information
    # ======================================================

    best_epoch = np.argmin(
        history.history["val_loss"]
    ) + 1

    best_val_loss = min(
        history.history["val_loss"]
    )

    print(
        f"Training Epochs : {len(history.history['loss'])}"
    )

    print(
        f"Best Epoch      : {best_epoch}"
    )

    print(
        f"Best Val Loss   : {best_val_loss:.4f}"
    )

    # ======================================================
    # Training / Validation Loss
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
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.show()

    # ======================================================
    # Training / Validation MAE
    # ======================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["mae"],
        label="Training MAE"
    )

    plt.plot(
        history.history["val_mae"],
        label="Validation MAE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE (kg)")
    plt.title("Training vs Validation MAE")

    plt.legend()
    plt.grid(True)

    plt.show()

    # ======================================================
    # Evaluate ANN Model
    # ======================================================
    
    loss, mae = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # ======================================================
    # Predict Weight
    # ======================================================
    
    predictions = model.predict(X_test)

    # ======================================================
    # Calculate RMSE
    # ======================================================
    
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # ======================================================
    # Calculate R²
    # ======================================================
    
    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"Test Loss : {loss:.4f}")
    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test R²   : {r2:.4f}")

    # ======================================================
    # Prediction Results
    # ======================================================
    
    prediction_df = pd.DataFrame({
        "Image": image_test.values,
        "Actual": y_test.values,
        "Predicted": predictions.flatten()
    })

    prediction_df["Error"] = (
        prediction_df["Predicted"]
        - prediction_df["Actual"]
    )

    prediction_df["Absolute Error"] = (
        prediction_df["Error"].abs()
    )


    # Round values
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
    # Save Model and Scaler
    # ======================================================

    model_path = (
        ANN_SPLIT_PATH
        / "ann_split.keras"
    )

    scaler_path = (
        ANN_SPLIT_PATH
        / "ann_split_scaler.joblib"
    )

    model.save(
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Architecture",
            "Input Features",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output",
            "Total Parameters",
            "Learning Rate",
            "Batch Size",
            "Maximum Epochs",
            "Training Epochs",
            "Best Epoch",
            "Best Validation Loss",
            "Target",
            "Train Images",
            "Validation Images",
            "Test Images",
            "Random State"
        ],
        "Value": [
            "Artificial Neural Network (ANN)",
            f"{len(FEATURES)}-{HIDDEN_LAYER_1}-{HIDDEN_LAYER_2}-1",
            len(FEATURES),
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            1,
            model.count_params(),
            LEARNING_RATE,
            BATCH_SIZE,
            EPOCHS,
            len(history.history["loss"]),
            best_epoch,
            best_val_loss,
            "Weight",
            len(X_train),
            len(X_val),
            len(X_test),
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
    # Metrics Information
    # ======================================================

    metrics_df = pd.DataFrame({
        "Dataset": ["Test"],
        "Loss": [loss],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2]
    })

    # ======================================================
    # Save Results
    # ======================================================

    results_path = (
        ANN_SPLIT_PATH
        / "ann_split_results.xlsx"
    )

    with pd.ExcelWriter(results_path) as writer:

        metrics_df.to_excel(
            writer,
            sheet_name="Metrics",
            index=False
        )

        prediction_df.to_excel(
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
    print("ANN REGRESSION - SPLIT RESULT")
    print("=" * 60)
    print(f"Training Epochs    : {len(history.history['loss'])}")
    print(f"Best Epoch         : {best_epoch}")
    print(f"Best Val Loss      : {best_val_loss:.4f}")
    print()
    print(f"Test Loss          : {loss:.4f}")
    print(f"Test MAE           : {mae:.4f} kg")
    print(f"Test RMSE          : {rmse:.4f} kg")
    print(f"Test R²            : {r2:.4f}")
    print()
    print(f"บันทึกโมเดลไว้ที่      : {model_path}")
    print(f"บันทึก Scaler ไว้ที่   : {scaler_path}")
    print(f"บันทึกผลลัพธ์ไว้ที่     : {results_path}")
    print("=" * 60)

# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()