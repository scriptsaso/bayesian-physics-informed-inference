#app/main.py
import argparse
import numpy as np
from app.data import load_data
from app.features import prepare_features
from app.model_bayes import build_model, train_bayesian, load_trace, generate_ppc
from app.plots import plot_ppc_arviz, run_model_diagnostics, plot_energy
from app.interpretation import identify_best_film, plot_shap_best_film, extract_posterior_means, plot_global_shap

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "load",  "ppc"])

    args = parser.parse_args()

    df = load_data(args.data)
    X_morph, X_stim, y, df_clean, morph_cols, stim_cols = prepare_features(df)

    if args.mode == "train":
        model = build_model(X_morph, X_stim, y)
        train_bayesian(model)

    elif args.mode == "ppc":

        trace_path = "artifacts/model_physical_trace_v2.nc"
        trace = load_trace(trace_path)

        model = build_model(X_morph, X_stim, y)

        ppc = generate_ppc(model, trace)

        plot_ppc_arviz(ppc)
        run_model_diagnostics(trace, ppc, y)
        plot_energy(trace)

        α_mean, coef_mean = extract_posterior_means(trace)

        best_idx, best_value = identify_best_film(
            α_mean,
            coef_mean,
            X_morph,
            X_stim,
            df_clean
        )

        # SHAP
        X_full = np.hstack([X_morph, X_stim])
        feature_names = morph_cols + stim_cols

        plot_shap_best_film(
            α_mean,
            coef_mean,
            X_full,
            feature_names,
            best_idx
        )

        plot_global_shap(
            trace,
            X_morph,
            X_stim,
            feature_names
        )

if __name__ == "__main__":
    main()