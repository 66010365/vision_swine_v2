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

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
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
    VALIDATION_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    DEBUG,
    POLYNOMIAL_ANN_LOGO_PATH
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

    POLYNOMIAL_ANN_LOGO_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_models_path = (
        POLYNOMIAL_ANN_LOGO_PATH / "fold_models"
    )

    fold_models_path.mkdir(
        parents=True,
        exist_ok=True
    )


    # ======================================================
    # Read ANN Dataset
    # ======================================================

    ann_df = pd.read_excel(
        MODEL_DATASET_PATH
    )

    required_columns = FEATURES + [
        "Pig_ID",
        "Weight"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in ann_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    ann_df = ann_df.dropna(
        subset=required_columns
    ).reset_index(drop=True)


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

        plt.title(
            f"{feature} vs Weight"
        )

        plt.xlabel(feature)
        plt.ylabel("Weight")

        plt.grid(True)

        plt.show()


    # ======================================================
    # Input Features
    # ======================================================

    X = ann_df[FEATURES]


    # ======================================================
    # Target
    # ======================================================

    y = ann_df["Weight"]


    # ======================================================
    # Image Names
    # ======================================================

    image_names = ann_df["Image"]


    # ======================================================
    # Pig Groups
    # ======================================================

    groups = ann_df["Pig_ID"]

    pig_ids = sorted(
        groups.unique()
    )


    print()
    print("=" * 60)
    print("POLYNOMIAL ANN REGRESSION - LOGO")
    print("=" * 60)
    print()

    print("Dataset")
    print("-" * 60)
    print(f"Total Images       : {len(ann_df):,}")
    print(f"Total Pigs         : {len(pig_ids)}")
    print(f"Number of Folds    : {len(pig_ids)}")


    # ======================================================
    # LOGO Results
    # ======================================================

    fold_results = []
    prediction_results = []


    # ======================================================
    # Leave-One-Group-Out
    # ======================================================

    for fold_number, test_pig in enumerate(
        pig_ids,
        start=1
    ):

        print()
        print("=" * 60)
        print(
            f"Fold {fold_number:02d}/{len(pig_ids)}"
        )
        print("=" * 60)

        print(
            f"Test Pig : {test_pig}"
        )


        # ==================================================
        # Test Dataset
        # ==================================================

        test_mask = (
            groups.values == test_pig
        )

        train_mask = ~test_mask

        X_test = X.iloc[test_mask].copy()
        y_test = y.iloc[test_mask].copy()

        image_test = image_names.iloc[test_mask].copy()

        X_train_all = X.iloc[train_mask].copy()
        y_train_all = y.iloc[train_mask].copy()

        groups_train_all = groups.iloc[train_mask].copy()


        # ==================================================
        # Split Train / Validation by Pig Group
        # ==================================================

        group_split = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE
        )

        train_index, validation_index = next(
            group_split.split(
                X_train_all,
                y_train_all,
                groups=groups_train_all
            )
        )

        X_train = X_train_all.iloc[train_index].copy()
        y_train = y_train_all.iloc[train_index].copy()

        X_val = X_train_all.iloc[validation_index].copy()
        y_val = y_train_all.iloc[validation_index].copy()


        # ==================================================
        # Polynomial Features
        # ==================================================

        polynomial = PolynomialFeatures(
            degree=2,
            include_bias=False
        )

        X_train = polynomial.fit_transform(
            X_train
        )

        X_val = polynomial.transform(
            X_val
        )

        X_test = polynomial.transform(
            X_test
        )


        # ==================================================
        # Normalize Features
        # ==================================================

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


        # ==================================================
        # Build ANN Model
        # ======================================================

        model = Sequential()

        model.add(
            Input(
                shape=(X_train.shape[1],)
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


        # ==================================================
        # Compile ANN Model
        # ==================================================

        model.compile(
            optimizer=Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="mse",
            metrics=["mae"]
        )


        # ==================================================
        # Early Stopping
        # ==================================================

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True
        )


        # ==================================================
        # Train ANN Model
        # ==================================================

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


        # ==================================================
        # Training Information
        # ==================================================

        best_epoch = (
            np.argmin(
                history.history["val_loss"]
            ) + 1
        )

        best_val_loss = min(
            history.history["val_loss"]
        )

        training_epochs = (
            len(history.history["loss"])
        )

        print(
            f"Training Epochs : {training_epochs}"
        )

        print(
            f"Best Epoch      : {best_epoch}"
        )

        print(
            f"Best Val Loss   : {best_val_loss:.4f}"
        )


        # ==================================================
        # Evaluate ANN Model
        # ==================================================

        loss, mae = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )


        # ==================================================
        # Predict Weight
        # ==================================================

        predictions = model.predict(
            X_test,
            verbose=0
        ).flatten()


        # ==================================================
        # Calculate RMSE
        # ==================================================

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )


        # ==================================================
        # Calculate R²
        # ==================================================

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


        # ==================================================
        # Prediction Results
        # ==================================================

        fold_prediction_df = pd.DataFrame({
            "Fold": fold_number,
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


        # ==================================================
        # Save Fold Model / Scaler / Polynomial
        # ==================================================

        fold_path = (
            fold_models_path
            / f"fold_{fold_number:02d}"
        )

        fold_path.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            fold_path / "model.keras"
        )

        scaler_path = (
            fold_path / "scaler.joblib"
        )

        polynomial_path = (
            fold_path / "polynomial.joblib"
        )

        model.save(
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        joblib.dump(
            polynomial,
            polynomial_path
        )


        # ==================================================
        # Fold Result
        # ==================================================

        fold_results.append({
            "Fold": fold_number,
            "Test Pig": test_pig,
            "Train Images": len(X_train),
            "Validation Images": len(X_val),
            "Test Images": len(X_test),
            "Test Loss": loss,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Training Epochs": training_epochs,
            "Best Epoch": best_epoch,
            "Best Validation Loss": best_val_loss,
            "Model Path": str(model_path),
            "Scaler Path": str(scaler_path),
            "Polynomial Path": str(polynomial_path)
        })


    # ======================================================
    # Combine Results
    # ======================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    prediction_results_df = pd.concat(
        prediction_results,
        ignore_index=True
    )


    # ======================================================
    # Overall Metrics
    # ======================================================

    average_mae = (
        fold_results_df["MAE"].mean()
    )

    average_rmse = (
        fold_results_df["RMSE"].mean()
    )

    overall_mae = (
        prediction_results_df["Absolute Error"].mean()
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            prediction_results_df["Actual"],
            prediction_results_df["Predicted"]
        )
    )

    overall_r2 = r2_score(
        prediction_results_df["Actual"],
        prediction_results_df["Predicted"]
    )

    average_best_epoch = (
        fold_results_df["Best Epoch"].mean()
    )


    # ======================================================
    # Model Information
    # ======================================================

    polynomial_feature_count = (
        len(FEATURES)
        + (
            len(FEATURES)
            * (len(FEATURES) + 1)
            // 2
        )
    )

    model_info_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Architecture",
            "Original Input Features",
            "Polynomial Degree",
            "Polynomial Features",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output",
            "Total Folds",
            "Learning Rate",
            "Batch Size",
            "Maximum Epochs",
            "Early Stopping Patience",
            "Random State",
            "Validation Method",
            "Fold Models Path"
        ],
        "Value": [
            "Artificial Neural Network with Polynomial Features",
            f"{polynomial_feature_count}-{HIDDEN_LAYER_1}-{HIDDEN_LAYER_2}-1",
            len(FEATURES),
            2,
            polynomial_feature_count,
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            1,
            len(pig_ids),
            LEARNING_RATE,
            BATCH_SIZE,
            EPOCHS,
            50,
            RANDOM_STATE,
            "GroupShuffleSplit",
            str(fold_models_path)
        ]
    })


    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Parameter": [
            "Scaler",
            "Scaler Fit",
            "Scaler File"
        ],
        "Value": [
            "MinMaxScaler",
            "Training Data Only",
            "One scaler per LOGO fold"
        ]
    })


    # ======================================================
    # Polynomial Information
    # ======================================================

    polynomial_info_df = pd.DataFrame({
        "Parameter": [
            "Transformer",
            "Degree",
            "Include Bias",
            "Original Features",
            "Polynomial Features",
            "Polynomial File"
        ],
        "Value": [
            "PolynomialFeatures",
            2,
            False,
            len(FEATURES),
            polynomial_feature_count,
            "One polynomial transformer per LOGO fold"
        ]
    })


    # ======================================================
    # Metrics Information
    # ======================================================

    metrics_df = pd.DataFrame({
        "Metric": [
            "Number of Pigs",
            "Number of Images",
            "Polynomial Degree",
            "Number of Polynomial Features",
            "Average Best Epoch",
            "Average MAE (kg)",
            "Average RMSE (kg)",
            "Overall MAE (kg)",
            "Overall RMSE (kg)",
            "Overall R²"
        ],
        "Value": [
            len(pig_ids),
            len(ann_df),
            2,
            polynomial_feature_count,
            average_best_epoch,
            average_mae,
            average_rmse,
            overall_mae,
            overall_rmse,
            overall_r2
        ]
    })


    # ======================================================
    # Save Results
    # ======================================================

    results_path = (
        POLYNOMIAL_ANN_LOGO_PATH
        / "polynomial_ann_logo_results.xlsx"
    )

    with pd.ExcelWriter(
        results_path
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

        prediction_results_df.to_excel(
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
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("POLYNOMIAL ANN REGRESSION - LOGO RESULT")
    print("=" * 60)

    print(
        f"Number of Pigs       : {len(pig_ids)}"
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
        f"บันทึก Fold Models ไว้ที่ : {fold_models_path}"
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
