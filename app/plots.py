#app/plots.py
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_ppc_arviz(ppc, save_path="artifacts/ppc_zoomed.png"):

    os.makedirs("artifacts", exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 2.5))

    az.plot_ppc(
        ppc,
        ax=ax,
        mean=True,
        alpha=0.6
    )

    ax.set_ylim(0, None)
    ax.set_xlabel("Observed S/V (normalized)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)

    ax.tick_params(axis='both', which='major',
                   labelsize=10, length=2, width=0.5)

    leg = ax.legend(fontsize=8, frameon=False,
                    loc="upper left", handlelength=1.5)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print("ArviZ PPC plot saved to:", save_path)

def run_model_diagnostics(trace, ppc, y, save_dir="artifacts"):

    os.makedirs(save_dir, exist_ok=True)

    # ======================================================
    # 1 LOO Cross Validation
    # ======================================================
    loo = az.loo(trace, pointwise=True)
    print(loo)
    fig, ax = plt.subplots(figsize=(3, 2.5))

    az.plot_khat(loo, ax=ax)

    # --- Axis formatting ---
    ax.set_xlabel("Data Point", fontsize=10)
    ax.set_ylabel("Shape parameter k", fontsize=10)
    ax.set_ylim(0, 1.05)
    # --- Ekseni ayarla ---
    ax.set_xlim(-0.5, 12.5)              # 0 ve 12’ye görsel pay verir
    ax.set_xticks(np.arange(0, 13, 2))

    # --- Tick formatting ---
    ax.tick_params(axis='both', which='major', labelsize=10, length=3, width=0.5)

    # --- Layout ---
    plt.tight_layout(pad=0.8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loo_pareto_k.png", dpi=300)
    plt.close()

    # ======================================================
    # 2 Residual Analysis
    # ======================================================
    y_pred = ppc.posterior_predictive["y_obs"].mean(
        dim=("chain", "draw")
    ).values

    residuals = y - y_pred

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

    sns.histplot(residuals, bins=8, kde=True, ax=ax[0])
    ax[0].set_title("Residual Distribution")

    ax[1].scatter(y_pred, residuals)
    ax[1].axhline(0, linestyle="--")
    ax[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/residual_diagnostics.png", dpi=300)
    plt.close()

    print("Diagnostics saved to:", save_dir)


def plot_energy(trace, save_path="artifacts/energy.png"):

    os.makedirs("artifacts", exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 2.5))

    az.plot_energy(
        trace,
        ax=ax,
        fill_alpha=[0.6, 0.6],
        fill_color=["#1f77b4", "#8c564b"],
        bw=0.3
    )

    # Axis styling
    ax.set_xlabel("Energy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(axis='both', labelsize=9, width=0.6, length=3)

    # Legend styling
    ax.legend(
        fontsize=9,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.8, 0.98),
        labelspacing=0.5,
        handlelength=1.5,
        handletextpad=0.6,
        borderpad=0.6
    )

    plt.subplots_adjust(left=0.05, right=1.5, top=0.99, bottom=0.0)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print("Energy diagnostic saved to:", save_path)