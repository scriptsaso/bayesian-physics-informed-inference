#app/features.py
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_features(df):

    # ===============================
    # Column definitions
    # ===============================
    morph_cols = [
        "I_x100","FWHM_x100","d_x100",
        "I_z100","FWHM_z100","d_z100",
        "I_z300","FWHM_z300","d_z300",
        "I_pi","FWHM_pi","d_pi",
        "Ratio_total","Porod_exponent","Porod_prefactor"
    ]

    stim_cols = ["Efield_num","Tg_num","CB_length_nm"]
    target_col = "S_V_porod"

    # ===============================
    # Clean + Filter
    # ===============================
    df = df.dropna(subset=morph_cols + stim_cols + [target_col]).reset_index(drop=True)

    df["Tg_num"] = df["Tg_num"].astype(str).astype(float).astype(int)
    df = df[df["Tg_num"] == 0].reset_index(drop=True)

    print(f"Removed Tg-applied films → Remaining samples: {len(df)}")

    # ===============================
    # Standardize predictors
    # ===============================
    scaler = StandardScaler().fit(df[morph_cols + stim_cols])
    X_all = scaler.transform(df[morph_cols + stim_cols])

    X_morph = X_all[:, :len(morph_cols)]
    X_stim  = X_all[:, len(morph_cols):]

    target_mean = df[target_col].mean()
    target_std = df[target_col].std()
    y = (df[target_col] - target_mean) / target_std 
    

    assert X_morph.shape[1] == len(morph_cols)
    assert X_stim.shape[1] == len(stim_cols)

    print(f"Feature shapes: {X_morph.shape[1]} morph | {X_stim.shape[1]} stim")

    return X_morph, X_stim, y, df, morph_cols, stim_cols


