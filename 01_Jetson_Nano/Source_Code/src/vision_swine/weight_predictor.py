# ======================================================
# Import Libraries
# ======================================================

import sys
from functools import lru_cache

import numpy as np
import joblib


# ======================================================
# Import Configuration
# ======================================================

from .config import (
    REGRESSION_MODEL_PATH,
    REGRESSION_SCALER_PATH,
)


# ======================================================
# NumPy Compatibility
# ======================================================

def _enable_numpy_compatibility():
    """
    Add compatibility aliases for NumPy module paths used by
    some previously saved model files.
    """

    import numpy.core as numpy_core

    sys.modules.setdefault("numpy._core", numpy_core)

    for name, module in list(sys.modules.items()):
        if name.startswith("numpy.core."):
            alias = name.replace("numpy.core.", "numpy._core.", 1)
            sys.modules.setdefault(alias, module)


# ======================================================
# Load Regression Model and Scaler
# ======================================================

@lru_cache(maxsize=1)
def _load_models():
    """
    Load the regression model and scaler once.

    Validate that both artifacts expect eight input features.
    """

    _enable_numpy_compatibility()

    model = joblib.load(str(REGRESSION_MODEL_PATH))
    scaler = joblib.load(str(REGRESSION_SCALER_PATH))

    expected = getattr(scaler, "n_features_in_", None)
    model_features = getattr(model, "n_features_in_", None)

    if expected != 8 or model_features != 8:
        raise ValueError(
            "Expected 8 input features, "
            f"but scaler={expected}, model={model_features}"
        )

    return model, scaler


# ======================================================
# Predict Pig Weight
# ======================================================

def predict_weight(features):
    """
    Estimate one pig's weight in kilograms from eight features.

    Accept either a one-dimensional list/array or a two-dimensional
    array containing one or more rows of features. Return the first
    predicted value as a float, matching the original behavior.
    """

    model, scaler = _load_models()

    X = np.asarray(features, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2 or X.shape[1] != 8:
        raise ValueError(
            f"Expected 8 features; received shape {X.shape}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("Features contain invalid numeric values")

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return float(np.asarray(prediction).reshape(-1)[0])
