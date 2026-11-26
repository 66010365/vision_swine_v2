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
    DEBUG
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ======================================================
# Main
# ======================================================

def main():

    # Read dataset
    model_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    if DEBUG:
        print(model_df.head())

    # ======================================================
    # Dataset Information
    # ======================================================
    
    if DEBUG:
        print(f"Total Images : {len(model_df)}")

    if DEBUG:
        print("\nImages per Pig")
        print(
            model_df["Pig_ID"].value_counts()
        )

    # ======================================================
    # Prepare Features and Target
    # ======================================================

    # Input Features
    X = model_df[FEATURES]

    # Target
    y = model_df["Weight"]

    # Image names
    image_names = model_df["Image"]

    if DEBUG:
        print(X.head())

    if DEBUG:
        print(y.head())

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
        
    if DEBUG:
        print(f"Train : {len(X_train)}")
        
    if DEBUG:
        print(f"Test  : {len(X_test)}")

    # ======================================================
    # Normalize Features
    # ======================================================

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    if DEBUG:
        print(X_train[:5])

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
    # Prediction
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

    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test R²   : {r2:.4f}")

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
# Run Program
# ======================================================

if __name__ == "__main__":
    main()