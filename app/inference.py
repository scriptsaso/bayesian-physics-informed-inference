#app/inference.py 
import joblib
import numpy as np

# ---- Load saved artifacts once ----
params = joblib.load("artifacts/posterior.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
stats  = joblib.load("artifacts/target_stats.pkl")

alpha = params["alpha"]
beta  = params["beta"]
gamma = params["gamma"]

target_mean = stats["mean"]
target_std  = stats["std"]

N_MORPH = 15  # morph feature count


def predict(raw_features):
    """
    raw_features: list of all features in correct order
    Returns REAL S/V value (not normalized)
    """

    X = scaler.transform([raw_features])

    X_morph = X[:, :N_MORPH]
    X_stim  = X[:, N_MORPH:]

    # Normalized prediction
    y_norm = alpha + X_morph @ beta + X_stim @ gamma

    # Convert back to real S/V
    y_real = y_norm * target_std + target_mean

    return float(y_real.squeeze())