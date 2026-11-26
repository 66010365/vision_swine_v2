# ======================================================
# Import Libraries
# ======================================================

import random
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
    DEBUG
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

    # Read ANN dataset
    ann_df = pd.read_excel(
        MODEL_DATASET_PATH
    )
    if DEBUG:
        print(ann_df.head())

    # ======================================================
    # Exploratory Data Analysis (EDA)
    # ======================================================

    # Correlation Analysis
    correlation = ann_df.corr(
        numeric_only=True
    )
    if DEBUG:
        print(correlation["Weight"])

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

    if DEBUG:
        print(X.head())

    if DEBUG:
        print(y.head())

    if DEBUG:
        print(y.dtype)

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

    if DEBUG:
        print(f"Train      : {len(X_train)}")
        print(f"Validation : {len(X_val)}")
        print(f"Test       : {len(X_test)}")

        print("\nTrain Weight")
        print(y_train)

        print("\nValidation Weight")
        print(y_val)

        print("\nTest Weight")
        print(y_test)

    # ======================================================
    # Normalize Features
    # ======================================================
    
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)

    if DEBUG:
        print(X_train[:5])


    # ======================================================
    # Build ANN Model
    # ======================================================
    
    model = Sequential()

    model.add(
    Dense(
        HIDDEN_LAYER_1,
        activation="relu",
        input_shape=(len(FEATURES),)
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
    if DEBUG:
        model.summary()

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

    if DEBUG:
        print(prediction_df)

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
# Run Program
# ======================================================

if __name__ == "__main__":
    main()