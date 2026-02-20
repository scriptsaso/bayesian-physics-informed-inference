import argparse
import os
import joblib
import arviz as az
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    # ------------------------------------------------------
    # 1 Load training data (ONLY for scaler + target stats)
    # ------------------------------------------------------
    df = pd.read_csv(args.data)

    morph_cols = [
        "I_x100","FWHM_x100","d_x100",
        "I_z100","FWHM_z100","d_z100",
        "I_z300","FWHM_z300","d_z300",
        "I_pi","FWHM_pi","d_pi",
        "Ratio_total","Porod_exponent","Porod_prefactor"
    ]

    stim_cols = ["Efield_num","Tg_num","CB_length_nm"]
    target_col = "S_V_porod"

    df = df.dropna(subset=morph_cols + stim_cols + [target_col]).reset_index(drop=True)
    df["Tg_num"] = df["Tg_num"].astype(str).astype(float).astype(int)
    df = df[df["Tg_num"] == 0].reset_index(drop=True)

    # ------------------------------------------------------
    # 2 Create scaler
    # ------------------------------------------------------
    scaler = StandardScaler().fit(df[morph_cols + stim_cols])
    joblib.dump(scaler, "artifacts/scaler.pkl")

    # ------------------------------------------------------
    # 3 Target normalization stats
    # ------------------------------------------------------
    target_mean = df[target_col].mean()
    target_std = df[target_col].std()

    joblib.dump(
        {"mean": target_mean, "std": target_std},
        "artifacts/target_stats.pkl"
    )

    # ------------------------------------------------------
    # 4 Extract posterior means from selected trace
    # ------------------------------------------------------
    trace = az.from_netcdf(args.trace)

    beta_mean = trace.posterior["β"].mean(dim=["chain","draw"]).values.flatten()
    gamma_mean = trace.posterior["γ"].mean(dim=["chain","draw"]).values.flatten()
    alpha_mean = trace.posterior["α"].mean(dim=["chain","draw"]).values.item()

    joblib.dump(
        {
            "alpha": alpha_mean,
            "beta": beta_mean,
            "gamma": gamma_mean
        },
        "artifacts/posterior.pkl"
    )

    print("  Freeze complete.")
    print("   scaler.pkl")
    print("   target_stats.pkl")
    print("   posterior.pkl")
    print("Generated from:", args.trace)


if __name__ == "__main__":
    main()