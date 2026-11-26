# ======================================================
# Import Libraries
# ======================================================

import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ======================================================
# Import Configuration
# ======================================================

from config import (
    MODEL_DATASET_PATH,
    FEATURES,
    RANDOM_STATE,
    KFOLD_SPLITS,
    VALIDATION_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    DEBUG,
    POLYNOMIAL_ANN_KFOLD_PATH
)


# ======================================================
# Main
# ======================================================

def main():

    # Set random seed
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # Create output folder
    POLYNOMIAL_ANN_KFOLD_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create fold model folder
    fold_models_path = (
        POLYNOMIAL_ANN_KFOLD_PATH
        / "fold_models"
    )

    fold_models_path.mkdir(
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
        print("=" * 60)
        print("POLYNOMIAL ANN REGRESSION - K-FOLD")
        print("=" * 60)
        print(
            f"Total Images : {len(model_df):,}"
        )
        print(
            f"Total Pigs   : "
            f"{model_df['Pig_ID'].nunique():,}"
        )
        print(
            f"K-Fold       : {KFOLD_SPLITS}"
        )
        print(
            "Polynomial Degree : 2"
        )


    # ======================================================
    # Prepare Features and Target
    # ======================================================

    # Input Features
    X = model_df[FEATURES]

    # Target
    y = model_df["Weight"]

    # Pig ID for Group K-Fold
    groups = model_df["Pig_ID"]

    # Image names
    image_names = model_df["Image"]


    # ======================================================
    # K-Fold Split
    # ======================================================

    group_kfold = GroupKFold(
        n_splits=KFOLD_SPLITS
    )

    fold_results = []
    prediction_results = []

    fold_mae = []
    fold_rmse = []
    fold_r2 = []
    best_epochs = []


    # ======================================================
    # Train and Test Each Fold
    # ======================================================

    for fold_number, (train_index, test_index) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        print()
        print("=" * 60)
        # Reset TensorFlow state and random seed for each fold
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_STATE + fold_number)

        print(
            f"FOLD {fold_number:02d}/{KFOLD_SPLITS}"
        )
        print("=" * 60)


        # ==================================================
        # Prepare Train/Test Data
        # ==================================================

        X_train_all = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train_all = y.iloc[train_index]
        y_test = y.iloc[test_index]

        image_test = image_names.iloc[test_index]
        pig_test = groups.iloc[test_index]

        groups_train_all = groups.iloc[train_index]


        if DEBUG:
            print(
                f"Train Images : "
                f"{len(X_train_all):,}"
            )
            print(
                f"Test Images  : "
                f"{len(X_test):,}"
            )
            print(
                f"Train Pigs   : "
                f"{groups_train_all.nunique():,}"
            )
            print(
                f"Test Pigs    : "
                f"{pig_test.nunique():,}"
            )


        # ==================================================
        # Validation Split
        # ==================================================

        validation_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE
        )

        train_inner_index, valid_inner_index = next(
            validation_splitter.split(
                X_train_all,
                y_train_all,
                groups_train_all
            )
        )

        X_train = X_train_all.iloc[
            train_inner_index
        ]

        X_valid = X_train_all.iloc[
            valid_inner_index
        ]

        y_train = y_train_all.iloc[
            train_inner_index
        ]

        y_valid = y_train_all.iloc[
            valid_inner_index
        ]


        if DEBUG:
            print(
                f"Train       : "
                f"{len(X_train):,}"
            )
            print(
                f"Validation  : "
                f"{len(X_valid):,}"
            )


        # ==================================================
        # Normalize Features
        # ==================================================

        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_valid_scaled = scaler.transform(
            X_valid
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ==================================================
        # Polynomial Transformation
        # ==================================================

        polynomial = PolynomialFeatures(
            degree=2,
            include_bias=False
        )

        X_train_poly = polynomial.fit_transform(
            X_train_scaled
        )

        X_valid_poly = polynomial.transform(
            X_valid_scaled
        )

        X_test_poly = polynomial.transform(
            X_test_scaled
        )


        # ==================================================
        # Build Polynomial ANN Model
        # ==================================================

        model = Sequential([
            Input(
                shape=(X_train_poly.shape[1],)
            ),

            Dense(
                HIDDEN_LAYER_1,
                activation="relu"
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


        # ==================================================
        # Compile Model
        # ==================================================

        model.compile(
            optimizer=Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="mse",
            metrics=["mae"]
        )


        # ==================================================
        # Train Model
        # ==================================================

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_poly,
            y_train,
            validation_data=(
                X_valid_poly,
                y_valid
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0
        )


        # ==================================================
        # Training Information
        # ==================================================

        training_epochs = len(
            history.history["loss"]
        )

        best_epoch = (
            np.argmin(
                history.history["val_loss"]
            )
            + 1
        )

        best_epochs.append(
            best_epoch
        )


        print(
            f"Training Epochs : "
            f"{training_epochs}"
        )

        print(
            f"Best Epoch      : "
            f"{best_epoch}"
        )


        # ==================================================
        # Test Prediction
        # ==================================================

        predictions = model.predict(
            X_test_poly,
            verbose=0
        ).flatten()

        if DEBUG:
            print(f"Prediction Min  : {predictions.min():.4f}")
            print(f"Prediction Max  : {predictions.max():.4f}")
            print(f"Prediction Mean : {predictions.mean():.4f}")
            print(f"Actual Mean     : {y_test.mean():.4f}")


        # ==================================================
        # Model Evaluation
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

        r2 = r2_score(
            y_test,
            predictions
        )

        fold_mae.append(
            mae
        )

        fold_rmse.append(
            rmse
        )

        fold_r2.append(
            r2
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
        # Fold Results
        # ==================================================

        fold_results.append({
            "Fold": fold_number,
            "Train Images": len(X_train),
            "Validation Images": len(X_valid),
            "Test Images": len(X_test),
            "Train Pigs": groups_train_all.iloc[
                train_inner_index
            ].nunique(),
            "Validation Pigs": groups_train_all.iloc[
                valid_inner_index
            ].nunique(),
            "Test Pigs": pig_test.nunique(),
            "Training Epochs": training_epochs,
            "Best Epoch": best_epoch,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


        # ==================================================
        # Prediction Results
        # ==================================================

        fold_prediction_df = pd.DataFrame({
            "Fold": fold_number,
            "Image": image_test.values,
            "Pig_ID": pig_test.values,
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

        prediction_results.append(
            fold_prediction_df
        )


        # ==================================================
        # Save Fold Model, Scaler and Polynomial
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
            fold_path
            / "model.keras"
        )

        scaler_path = (
            fold_path
            / "scaler.joblib"
        )

        polynomial_path = (
            fold_path
            / "polynomial.joblib"
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


        # Clear TensorFlow session
        tf.keras.backend.clear_session()


    # ======================================================
    # Combine Fold Results
    # ======================================================

    fold_results_df = pd.DataFrame(
        fold_results
    )

    prediction_result_df = pd.concat(
        prediction_results,
        ignore_index=True
    )


    # ======================================================
    # Overall Metrics
    # ======================================================

    average_mae = np.mean(
        fold_mae
    )

    average_rmse = np.mean(
        fold_rmse
    )

    average_r2 = np.mean(
        fold_r2
    )

    average_best_epoch = np.mean(
        best_epochs
    )

    overall_mae = mean_absolute_error(
        prediction_result_df["Actual"],
        prediction_result_df["Predicted"]
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            prediction_result_df["Actual"],
            prediction_result_df["Predicted"]
        )
    )

    overall_r2 = r2_score(
        prediction_result_df["Actual"],
        prediction_result_df["Predicted"]
    )


    metrics_df = pd.DataFrame({
        "Metric": [
            "Number of Images",
            "Number of Pigs",
            "K-Fold Splits",
            "Polynomial Degree",
            "Average Best Epoch",
            "Average MAE",
            "Average RMSE",
            "Average R2",
            "Overall MAE",
            "Overall RMSE",
            "Overall R2"
        ],

        "Value": [
            len(model_df),
            model_df["Pig_ID"].nunique(),
            KFOLD_SPLITS,
            2,
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
            "Original Features",
            "Polynomial Degree",
            "Polynomial Features",
            "Target",
            "K-Fold Method",
            "Group",
            "K-Fold Splits",
            "Random State",
            "Learning Rate",
            "Epochs",
            "Batch Size",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Activation",
            "Early Stopping Patience"
        ],

        "Value": [
            "Polynomial Artificial Neural Network",
            len(FEATURES),
            2,
            polynomial_feature_count,
            "Weight",
            "GroupKFold",
            "Pig_ID",
            KFOLD_SPLITS,
            RANDOM_STATE,
            LEARNING_RATE,
            EPOCHS,
            BATCH_SIZE,
            HIDDEN_LAYER_1,
            HIDDEN_LAYER_2,
            "ReLU",
            50
        ]
    })


    # ======================================================
    # Scaler Information
    # ======================================================

    scaler_info_df = pd.DataFrame({
        "Feature": FEATURES
    })


    # ======================================================
    # Polynomial Information
    # ======================================================

    polynomial_info_df = pd.DataFrame({
        "Parameter": [
            "Polynomial Degree",
            "Include Bias",
            "Original Features",
            "Generated Features"
        ],

        "Value": [
            2,
            False,
            len(FEATURES),
            polynomial_feature_count
        ]
    })


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
    # Save Excel Results
    # ======================================================

    results_path = (
        POLYNOMIAL_ANN_KFOLD_PATH
        / "polynomial_ann_kfold_results.xlsx"
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
    # Final Summary
    # ======================================================

    print()
    print("=" * 60)
    print("สรุปผล Polynomial ANN Regression - K-Fold")
    print("=" * 60)

    print(
        f"Number of Images        : "
        f"{len(model_df):,} ภาพ"
    )

    print(
        f"Number of Pigs          : "
        f"{model_df['Pig_ID'].nunique():,} ตัว"
    )

    print(
        f"K-Fold                  : "
        f"{KFOLD_SPLITS} Folds"
    )

    print(
        f"Polynomial Degree       : 2"
    )

    print(
        f"Average Best Epoch      : "
        f"{average_best_epoch:.1f}"
    )

    print()
    print(
        f"Average MAE             : "
        f"{average_mae:.4f} kg"
    )

    print(
        f"Average RMSE            : "
        f"{average_rmse:.4f} kg"
    )

    print(
        f"Average R²              : "
        f"{average_r2:.4f}"
    )

    print(
        f"Overall MAE             : "
        f"{overall_mae:.4f} kg"
    )

    print(
        f"Overall RMSE            : "
        f"{overall_rmse:.4f} kg"
    )

    print(
        f"Overall R²              : "
        f"{overall_r2:.4f}"
    )

    print()
    print(
        f"บันทึก Fold Models ไว้ที่ : "
        f"{fold_models_path}"
    )

    print(
        f"บันทึกผลลัพธ์ไว้ที่       : "
        f"{results_path}"
    )

    print("=" * 60)


# ======================================================
# Run Program
# ======================================================

if __name__ == "__main__":
    main()
