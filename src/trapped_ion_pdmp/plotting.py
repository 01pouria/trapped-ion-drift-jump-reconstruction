from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig, path, dpi: int = 220):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_stochastic_structure(d2_table: pd.DataFrame, gamma_hat: float, gamma_theory: float):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    axes[0].plot(d2_table["tau"], d2_table["median_D2_trace"], marker="o")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("median Tr[D(2)]")
    axes[0].set_title("Finite-lag second KM coefficient")

    axes[1].plot(
        d2_table["tau"],
        d2_table["median_D2_trace_over_tau"],
        marker="o",
    )
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Tr[D(2)] / tau")
    axes[1].set_title("D(2)(tau) ~ O(tau)")

    axes[2].scatter([0], [gamma_hat], label="Recovered")
    axes[2].axhline(gamma_theory, linestyle="--", label="Theory")
    axes[2].set_xticks([0], ["gamma"])
    axes[2].set_ylabel("Jump rate")
    axes[2].set_title("Relaxation-rate recovery")
    axes[2].legend()

    fig.tight_layout()
    return fig


def plot_rate_recovery(parametric_table: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    eta = parametric_table[parametric_table["case"].isin(["eta_0.10", "baseline", "eta_0.20"])]
    axes[0].plot(eta["eta"], eta["g_hat"], marker="o", label="Recovered")
    axes[0].plot(eta["eta"], eta["g_true"], linestyle="--", label="Theory")
    axes[0].set_xlabel("eta")
    axes[0].set_ylabel("g")
    axes[0].set_title("g vs eta")
    axes[0].legend()

    omega = parametric_table[parametric_table["case"].isin(["Omega_0.70", "baseline", "Omega_1.30"])]
    axes[1].plot(omega["Omega"], omega["g_hat"], marker="o", label="Recovered")
    axes[1].plot(omega["Omega"], omega["g_true"], linestyle="--", label="Theory")
    axes[1].set_xlabel("Omega")
    axes[1].set_ylabel("g")
    axes[1].set_title("g vs Omega")

    t1 = parametric_table[parametric_table["case"].isin(["T1_50", "baseline", "T1_120"])].copy()
    t1["inv_T1"] = 1.0 / t1["T1"]
    axes[2].plot(t1["inv_T1"], t1["gamma_hat"], marker="o", label="Recovered")
    axes[2].plot(t1["inv_T1"], t1["gamma_true"], linestyle="--", label="Theory")
    axes[2].set_xlabel("1/T1")
    axes[2].set_ylabel("gamma")
    axes[2].set_title("gamma vs 1/T1")

    fig.tight_layout()
    return fig


def plot_ablation(table: pd.DataFrame):
    metrics = [
        "mean_NRMSE",
        "variance_NRMSE",
        "wasserstein",
        "survival_RMSE",
        "covariance_error",
    ]
    ax = table.set_index("model")[metrics].T.plot(kind="bar", figsize=(8.5, 4.3))
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Explicit-jump PDMP versus deterministic and Gaussian surrogates")
    ax.figure.tight_layout()
    return ax.figure
