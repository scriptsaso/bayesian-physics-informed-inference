#app/interpretation.py 
import numpy as np
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl


# ==========================================================
# 1 Extract posterior means (ONLY ONCE)
# ==========================================================

def extract_posterior_means(trace):

    β_mean = trace.posterior["β"].mean(dim=["chain", "draw"]).values.flatten()
    γ_mean = trace.posterior["γ"].mean(dim=["chain", "draw"]).values.flatten()
    α_mean = trace.posterior["α"].mean(dim=["chain", "draw"]).values.item()

    coef_mean = np.concatenate([β_mean, γ_mean])

    return α_mean, coef_mean


# ==========================================================
# 2 Identify best-performing film
# ==========================================================

def identify_best_film(
    α_mean,
    coef_mean,
    X_morph,
    X_stim,
    df_clean,
    save_path="artifacts/best_film.txt"
):

    X_full = np.hstack([X_morph, X_stim])
    y_pred = α_mean + X_full @ coef_mean

    best_idx = np.argmax(y_pred)
    best_value = y_pred[best_idx]

    print("🏆 Best-performing film:")
    print(df_clean.iloc[best_idx][
        ["system", "Efield", "Tg_applied", "CB_length_nm", "S_V_porod"]
    ])
    print("Predicted normalized S/V:", best_value)

    os.makedirs("artifacts", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Best-performing film\n")
        f.write(str(df_clean.iloc[best_idx]) + "\n")
        f.write(f"\nPredicted normalized S/V: {best_value}\n")

    return best_idx, best_value


# ==========================================================
# 3 SHAP explanation
# ==========================================================

def plot_shap_best_film(
    α_mean,
    coef_mean,
    X_full,
    feature_names,
    best_idx,
    save_path="artifacts/best_prm.png"
):

    os.makedirs("artifacts", exist_ok=True)

    shap_values = X_full[best_idx] * coef_mean

    sample_expl = shap.Explanation(
        values=shap_values,
        base_values=np.array([α_mean]),
        data=X_full[best_idx],
        feature_names=feature_names
    )

    mask = np.abs(sample_expl.values) > 0.05

    filtered_expl = shap.Explanation(
        values=sample_expl.values[mask],
        base_values=sample_expl.base_values,
        data=sample_expl.data[mask],
        feature_names=np.array(sample_expl.feature_names)[mask]
    )

    shap.plots.bar(filtered_expl, max_display=13, show=False, show_data=False)

    ax = plt.gca()

    clean_labels = [
        lbl.get_text().split('=')[0].strip()
        for lbl in ax.get_yticklabels()
    ]
    ax.set_yticklabels(clean_labels)

    ax.tick_params(axis='y', length=0, pad=10)
    plt.xlabel("Impact on normalized S/V$_{Porod}$", fontsize=12)

    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(left=0.32)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=600,
                bbox_inches="tight",
                transparent=False)
    plt.close()

    print("SHAP explanation saved to:", save_path)



def plot_global_shap(
    trace,
    X_morph,
    X_stim,
    feature_names,
    save_path="artifacts/global_shap.png"
):

    os.makedirs("artifacts", exist_ok=True)

    # --------------------------------------------------
    # 1. Posterior means
    # --------------------------------------------------
    β_mean = trace.posterior["β"].mean(dim=["chain", "draw"]).values.flatten()
    γ_mean = trace.posterior["γ"].mean(dim=["chain", "draw"]).values.flatten()

    coef_mean = np.concatenate([β_mean, γ_mean])

    # --------------------------------------------------
    # 2. Build full feature matrix
    # --------------------------------------------------
    X = np.hstack([X_morph, X_stim])

    # Linear SHAP values
    shap_values = X * coef_mean

    # --------------------------------------------------
    # 3. Custom academic colormap
    # --------------------------------------------------
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "academic",
        ["#3b4cc0", "#e6e6e6", "#b40426"],
        N=256
    )

    # --------------------------------------------------
    # 4. SHAP summary plot
    # --------------------------------------------------
    shap.summary_plot(
        shap_values,
        features=X,
        feature_names=feature_names,
        show=False,
        plot_type="violin",
        color=custom_cmap,
        max_display=20,
    )

    # --------------------------------------------------
    # 5. Academic formatting
    # --------------------------------------------------
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.set_dpi(600)
    ax = plt.gca()

    plt.xlabel(
        "SHAP value (impact on normalized S/V$_{Porod}$)",
        fontsize=10
    )

    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_rotation_mode('anchor')
        label.set_ha('right')
        label.set_va('center')
        label.set_fontsize(12)
        label.set_x(0.05)

    # Colorbar formatting
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("Feature value", fontsize=12,
                       rotation=270, labelpad=5)
    cbar_ax.tick_params(labelsize=10)
    cbar_ax.set_yticklabels(["Low", "High"],
                            fontsize=12, rotation=270)

    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=600,
                bbox_inches="tight",
                transparent=False)
    plt.close()

    print("Global SHAP plot saved to:", save_path)