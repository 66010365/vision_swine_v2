# ======================================================
# Import Libraries
# ======================================================

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
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
    DEBUG
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ======================================================
# Main
# ======================================================

def main():

    # ======================================================
    # Read dataset
    # ======================================================
    ann_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    if DEBUG:
        print(ann_df.head())

    # ======================================================
    # Dataset Information
    # ======================================================

    if DEBUG:
        print(f"Total Images : {len(ann_df)}")

    if DEBUG:
        print("\nImages per Pig")
        print(
            ann_df["Pig_ID"].value_counts()
        )

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

    mae_scores = []
    rmse_scores = []

    prediction_results = []
    logo_results = []

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

        # ==================================================
        # Normalize Features
        # ==================================================

        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

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

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        test_pig = groups.iloc[test_idx].iloc[0]

        logo_results.append({
            "Pig_ID": test_pig,
            "MAE": mae,
            "RMSE": rmse,
        })

        # ==================================================
        # Prediction Results
        # ==================================================

        prediction_df = pd.DataFrame({
            "Image": image_test.values,
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

    # ======================================================
    # LOGO Results per Pig
    # ======================================================

    logo_result_df = pd.DataFrame(
        logo_results
    )

    print()
    print("======================================================")
    print("LOGO Results per Pig")
    print("======================================================")

    print(logo_result_df)

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
    print("======================================================")
    print("LOGO Evaluation Summary")
    print("======================================================")

    print(
        f"LOGO Average MAE  : {mean_mae:.4f}"
    )

    print(
        f"LOGO Average RMSE : {mean_rmse:.4f}"
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
    # Prediction Results
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

    if DEBUG:
        print(prediction_df)

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

    plt.xlabel("Actual Weight (kg)")
    plt.ylabel("Predicted Weight (kg)")
    plt.title("Actual vs Predicted Weight")

    plt.legend()
    plt.grid(True)

    plt.show()

# ======================================================
# Run Program
# ======================================================
if __name__ == "__main__":
    main()