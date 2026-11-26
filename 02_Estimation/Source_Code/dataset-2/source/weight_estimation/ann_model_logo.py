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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
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
    VALIDATION_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    DEBUG,
    ANN_LOGO_PATH
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

    ANN_LOGO_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_models_path = ANN_LOGO_PATH / "fold_models"

    fold_models_path.mkdir(
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

    # Pig Groups
    groups = ann_df["Pig_ID"]

    # Image names
    image_names = ann_df["Image"]

    # ======================================================
    # ANN REGRESSION - LOGO
    # ======================================================

    print()
    print("=" * 60)
    print("ANN REGRESSION - LOGO")
    print("=" * 60)
    print()

    print("Dataset")
    print("-" * 60)
    print(f"Total Images       : {len(ann_df):,}")
    print(f"Total Pigs         : {groups.nunique():,}")
    print()

    # ======================================================
    # LOGO
    # ======================================================

    pig_ids = groups.drop_duplicates().tolist()

    fold_results = []
    prediction_results = []

    for fold, test_pig in enumerate(
        pig_ids,
        start=1
    ):

        print()
        print("=" * 60)
        print(f"Fold {fold:02d}/{len(pig_ids)}")
        print(f"Test Pig          : {test_pig}")
        print("=" * 60)

        # ======================================================
        # Leave-One-Group-Out Split
        # ======================================================

        test_mask = groups == test_pig

        X_train_full = X.loc[
            ~test_mask
        ]

        y_train_full = y.loc[
            ~test_mask
        ]

        groups_train_full = groups.loc[
            ~test_mask
        ]

        X_test = X.loc[
            test_mask
        ]

        y_test = y.loc[
            test_mask
        ]

        image_test = image_names.loc[
            test_mask
        ]

        # ======================================================
        # Split Train / Validation
        # ======================================================

        group_split = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE
        )

        train_index, val_index = next(
            group_split.split(
                X_train_full,
                y_train_full,
                groups=groups_train_full
            )
        )

        X_train = X_train_full.iloc[
            train_index
        ]

        y_train = y_train_full.iloc[
            train_index
        ]

        X_val = X_train_full.iloc[
            val_index
        ]

        y_val = y_train_full.iloc[
            val_index
        ]

        # ======================================================
        # Normalize Features
        # ======================================================

        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_val = scaler.transform(
            X_val
        )

        X_test = scaler.transform(
            X_test
        )

        # ======================================================
        # Build ANN Model
        # ======================================================

        tf.keras.backend.clear_session()

        model = Sequential()

        model.add(
            Input(
                shape=(len(FEATURES),)
            )
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

        best_epoch = (
            np.argmin(
                history.history["val_loss"]
            ) + 1
        )

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
        ).flatten()

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

        print(
            f"Test Loss : {loss:.4f}"
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

        # ======================================================
        # Save Model and Scaler
        # ======================================================

        fold_path = (
            fold_models_path
            / f"fold_{fold:02d}"
        )

        fold_path.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            fold_path
            / "model.keras"
        )

        scaler_path = (
            fold_path
            / "scaler.joblib"
        )

        model.save(
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        # ======================================================
        # Prediction Results
        # ======================================================

        fold_prediction_df = pd.DataFrame({
            "Fold": fold,
            "Pig_ID": test_pig,
            "Image": image_test.values,
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

        fold_prediction_df["Predicted"] = (
            fold_prediction_df["Predicted"]
            .round(2)
        )

        fold_prediction_df["Error"] = (
            fold_prediction_df["Error"]
            .round(2)
        )

        fold_prediction_df["Absolute Error"] = (
            fold_prediction_df["Absolute Error"]
            .round(2)
        )

        prediction_results.append(
            fold_prediction_df
        )

        # ======================================================
        # Fold Results
        # ======================================================

        fold_results.append({
            "Fold": fold,
            "Pig_ID": test_pig,
            "Train Images": len(X_train),
            "Validation Images": len(X_val),
            "Test Images": len(X_test),
            "Training Epochs": len(
                history.history["loss"]
            ),
            "Best Epoch": best_epoch,
            "Best Validation Loss": best_val_loss,
            "Test Loss": loss,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Model Path": str(model_path),
            "Scaler Path": str(scaler_path)
        })

    # ======================================================
    # Combine Results
    # ======================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    prediction_df = pd.concat(
        prediction_results,
        ignore_index=True
    )

    # ======================================================
    # Overall Metrics
    # ======================================================

    average_best_epoch = (
        fold_results_df["Best Epoch"]
        .mean()
    )

    average_mae = (
        fold_results_df["MAE"]
        .mean()
    )

    average_rmse = (
        fold_results_df["RMSE"]
        .mean()
    )

    average_r2 = (
        fold_results_df["R2"]
        .mean()
    )

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

    # ======================================================
    # Model Information
    # ======================================================

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Validation Method",
            "Group",
            "Architecture",
            "Input Features",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output",
            "Total Parameters",
            "Learning Rate",
            "Batch Size",
            "Maximum Epochs",
            "Early Stopping",
            "Early Stopping Patience",
            "Restore Best Weights",
            "Number of Folds",
            "Number of Pigs",
            "Number of Images",
            "Average Best Epoch",
            "Average MAE",
            "Average RMSE",
            "Average R2",
            "Overall MAE",
            "Overall RMSE",
            "Overall R2",
            "Random State"
        ],
        "Value": [
            "Artificial Neural Network (ANN)",
            "Leave-One-Group-Out (LOGO)",
            "Pig_ID",
            f"{len(FEATURES)}-{HIDDEN_LAYER_1}-{HIDDEN_LAYER_2}-1",
            len(FEATURES),
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            1,
            HIDDEN_LAYER_1 * len(FEATURES)
            + HIDDEN_LAYER_1
            + HIDDEN_LAYER_2 * HIDDEN_LAYER_1
            + HIDDEN_LAYER_2
            + HIDDEN_LAYER_2
            + 1,
            LEARNING_RATE,
            BATCH_SIZE,
            EPOCHS,
            "Yes",
            50,
            True,
            len(pig_ids),
            groups.nunique(),
            len(ann_df),
            average_best_epoch,
            average_mae,
            average_rmse,
            average_r2,
            overall_mae,
            overall_rmse,
            overall_r2,
            RANDOM_STATE
        ]
    })

    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Information": [
            "Scaler",
            "Scaling Method",
            "Number of Features",
            "Fitted Separately Per Fold",
            "Number of Saved Scalers",
            "Scaler Folder"
        ],
        "Value": [
            "MinMaxScaler",
            "Min-Max Normalization",
            len(FEATURES),
            True,
            len(pig_ids),
            str(fold_models_path)
        ]
    })

    # ======================================================
    # Metrics Information
    # ======================================================

    metrics_df = pd.DataFrame({
        "Metric": [
            "Average Best Epoch",
            "Average MAE (kg)",
            "Average RMSE (kg)",
            "Average R2",
            "Overall MAE (kg)",
            "Overall RMSE (kg)",
            "Overall R2"
        ],
        "Value": [
            average_best_epoch,
            average_mae,
            average_rmse,
            average_r2,
            overall_mae,
            overall_rmse,
            overall_r2
        ]
    })

    # ======================================================
    # Save Results
    # ======================================================

    results_path = (
        ANN_LOGO_PATH
        / "ann_logo_results.xlsx"
    )

    with pd.ExcelWriter(
        results_path,
        engine="openpyxl"
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
    print("ANN REGRESSION - LOGO RESULT")
    print("=" * 60)

    print(
        f"Number of Pigs       : {groups.nunique():,}"
    )

    print(
        f"Number of Images     : {len(ann_df):,}"
    )

    print(
        f"Average Best Epoch   : {average_best_epoch:.1f}"
    )

    print(
        f"Average MAE          : {average_mae:.4f} kg"
    )

    print(
        f"Average RMSE         : {average_rmse:.4f} kg"
    )

    print(
        f"Average R²           : {average_r2:.4f}"
    )

    print(
        f"Overall MAE          : {overall_mae:.4f} kg"
    )

    print(
        f"Overall RMSE         : {overall_rmse:.4f} kg"
    )

    print(
        f"Overall R²           : {overall_r2:.4f}"
    )

    print()

    print(
        f"บันทึก Model + Scaler ไว้ที่ : {fold_models_path}"
    )

    print(
        f"บันทึกผลลัพธ์ไว้ที่          : {results_path}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
