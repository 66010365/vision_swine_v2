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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ======================================================
# Import Configuration
# ======================================================

from config import (
    MODEL_DATASET_PATH,
    FEATURES,
    RANDOM_STATE,
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
    # Feature Statistics by Pig
    # ======================================================

    feature_summary = ann_df.groupby("Pig_ID")[
        FEATURES
    ].mean()

    print()
    print("======================================================")
    print("Feature Mean by Pig")
    print("======================================================")

    print(feature_summary)

    # ======================================================
    # Feature Range by Pig
    # ======================================================

    feature_range = ann_df.groupby("Pig_ID")[FEATURES].agg(
        ["min", "max"]
    )

    print()
    print("======================================================")
    print("Feature Range by Pig")
    print("======================================================")

    print(feature_range)

    # ======================================================
    # Correlation by Pig
    # ======================================================

    print()
    print("======================================================")
    print("Feature Correlation with Weight by Pig")
    print("======================================================")

    for pig_id, pig_df in ann_df.groupby("Pig_ID"):

        print()
        print(f"{pig_id}")

        print(
            pig_df[FEATURES + ["Weight"]].corr()["Weight"]
        )

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

    # ======================================================
    # Groups
    # ======================================================

    groups = ann_df["Pig_ID"]

    # Image names
    image_names = ann_df["Image"]

    # ======================================================
    # Leave-One-Group-Out (LOGO)
    # ======================================================

    logo = LeaveOneGroupOut()

    pig_scores = []
    mae_scores = []
    rmse_scores = []
    
    prediction_results = []

    for train_idx, test_idx in logo.split(
        X,
        y,
        groups
    ):

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

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        # ==================================================
        # Build ANN Model
        # ==================================================

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

        print(
            f"\nTraining Test Pig: {test_pig}"
        )
        
        # ==================================================
        # Train ANN Model
        # ==================================================

        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

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

        predictions = model.predict(
            X_test,
            verbose=0
        )

        # ======================================================
        # Calculate RMSE
        # ======================================================

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        print(
            f"Test Pig : {test_pig}"
        )

        print(
            f"Test Loss : {loss:.4f}"
        )

        print(
            f"Test MAE  : {mae:.4f}"
        )

        print(
            f"Test RMSE : {rmse:.4f}"
        )

        # ======================================================
        # Store LOGO Scores
        # ======================================================

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        pig_scores.append(test_pig)

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

        prediction_results.append(
            prediction_df
        )
    # ======================================================
    # Combine Prediction Results
    # ======================================================

    prediction_df = pd.concat(
        prediction_results,
        ignore_index=True
    )

    if DEBUG:
        print(prediction_df)

    # ======================================================
    # LOGO Pig Evaluation
    # ======================================================

    logo_summary = pd.DataFrame({
        "Pig_ID": pig_scores,
        "MAE": mae_scores,
        "RMSE": rmse_scores
    })

    print()
    print("======================================================")
    print("LOGO Pig Evaluation")
    print("======================================================")

    print(logo_summary)

    # ======================================================
    # Overall LOGO Evaluation
    # ======================================================

    overall_mae = np.mean(
        prediction_df["Absolute Error"]
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
    print("======================================================")
    print("Overall LOGO Evaluation")
    print("======================================================")

    print(
        f"Overall MAE  : {overall_mae:.4f}"
    )

    print(
        f"Overall RMSE : {overall_rmse:.4f}"
    )

    print(
        f"Overall R²   : {overall_r2:.4f}"
    )

# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()