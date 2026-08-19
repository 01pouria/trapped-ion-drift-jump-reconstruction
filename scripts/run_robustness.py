#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trapped_ion_pdmp.config import ACTIVE_STATE_COLUMNS
from trapped_ion_pdmp.km import (
    fit_kernel_neighbor_model,
    kernel_drift_predict,
    markov_memory_diagnostic,
    state_space_matrix_diagnostics,
)
from trapped_ion_pdmp.pipeline import (
    fit_pdmp_components,
    simulate_baseline,
    simulate_parametric_case,
)
from trapped_ion_pdmp.rates import integrated_exposure, recover_coupling, recover_gamma
from trapped_ion_pdmp.statistics import trajectory_cluster_bootstrap_rates
from trapped_ion_pdmp.validation import finite_lag_km_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated" / "robustness"


def recover_case(case, split_seed):
    coupling = recover_coupling(
        case["dataframe"],
        case["jump_times"],
        case["eta"],
        case["Omega"],
        case["cfg"].mode_weights,
        seed=split_seed,
    )
    gamma = recover_gamma(
        case["dataframe"],
        case["collapse_channels"],
        case["T1"],
        coupling["train_ids"],
        coupling["validation_ids"],
    )
    return coupling, gamma


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Generating baseline for diagnostics...")
    baseline = simulate_baseline()
    fitted = fit_pdmp_components(baseline)

    state_diag = state_space_matrix_diagnostics(
        baseline.dataframe,
        ACTIVE_STATE_COLUMNS,
    )
    pd.DataFrame(
        [
            {
                "dimension": state_diag["dimension"],
                "rank": state_diag["rank"],
                "effective_dimension": state_diag["effective_dimension"],
                "condition_number": state_diag["condition_number"],
            }
        ]
    ).to_csv(OUT / "state_space.csv", index=False)

    memory = markov_memory_diagnostic(
        baseline.dataframe,
        ACTIVE_STATE_COLUMNS,
        lag_steps=4,
    )
    pd.DataFrame(
        [
            {
                "NRMSE_present": memory["average_NRMSE_present"],
                "NRMSE_present_plus_past": memory["average_NRMSE_memory"],
                "memory_gain": memory["memory_gain"],
            }
        ]
    ).to_csv(OUT / "memory.csv", index=False)

    d2 = finite_lag_km_summary(
        baseline.dataframe,
        baseline.jump_times,
        fitted["train_ids"],
        fitted["validation_ids"],
        ACTIVE_STATE_COLUMNS,
    )
    d2.to_csv(OUT / "d2_finite_lag.csv", index=False)

    print("Running independent-seed robustness...")
    seed_rows = []
    for i, seed in enumerate((33346, 34346, 35346)):
        case = simulate_parametric_case(
            f"seed_{i}",
            0.15,
            1.0,
            80.0,
            ntraj=80,
            n_time=601,
            seed=seed,
        )
        coupling, gamma = recover_case(case, seed + 100)
        seed_rows.append(
            {
                "seed": seed,
                "g_hat": coupling["g_hat"],
                "g_relative_error": coupling["g_relative_error"],
                "gamma_hat": gamma["gamma_hat"],
                "gamma_relative_error": gamma["gamma_relative_error"],
                "train_jumps": gamma["train_jumps"],
                "validation_jumps": gamma["validation_jumps"],
            }
        )
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(OUT / "independent_seeds.csv", index=False)

    print("Running time-step robustness...")
    dt_rows = []
    for n_time in (401, 801, 1201):
        case = simulate_parametric_case(
            f"dt_{n_time}",
            0.15,
            1.0,
            80.0,
            ntraj=50,
            n_time=n_time,
            seed=37346,
        )
        coupling, gamma = recover_case(case, 38346)
        dt_rows.append(
            {
                "n_time": n_time,
                "dt": case["times"][1] - case["times"][0],
                "g_hat": coupling["g_hat"],
                "g_relative_error": coupling["g_relative_error"],
                "gamma_hat": gamma["gamma_hat"],
                "gamma_relative_error": gamma["gamma_relative_error"],
            }
        )
    pd.DataFrame(dt_rows).to_csv(OUT / "time_step.csv", index=False)

    print("Running kernel-neighbor sensitivity...")
    transitions = fitted["transitions"]
    val_mask = np.isin(
        transitions["trajectory"],
        fitted["validation_ids"],
    ) & ~transitions["jump"]
    X_val = transitions["X"][val_mask]
    Y_val = transitions["dX"][val_mask] / transitions["tau"]

    rng = np.random.default_rng(39346)
    idx = rng.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
    X_probe, Y_probe = X_val[idx], Y_val[idx]

    predictions = {}
    kernel_rows = []
    for neighbors in (200, 400, 800):
        model = fit_kernel_neighbor_model(
            fitted["X_drift_train"],
            neighbors,
        )
        pred = kernel_drift_predict(
            model,
            fitted["dX_drift_train"],
            X_probe,
            transitions["tau"],
        )
        D1 = pred["D1"]
        predictions[neighbors] = D1
        err = np.sqrt(np.mean((D1 - Y_probe) ** 2))
        scale = np.sqrt(np.mean(Y_probe**2))
        kernel_rows.append(
            {
                "neighbors": neighbors,
                "drift_NRMSE": err / max(scale, 1e-12),
                "median_support_radius": np.median(pred["neighbor_radius"]),
            }
        )

    reference = predictions[400]
    reference_scale = np.sqrt(np.mean(reference**2))
    for row in kernel_rows:
        diff = np.sqrt(np.mean((predictions[row["neighbors"]] - reference) ** 2))
        row["relative_change_vs_K400"] = diff / max(reference_scale, 1e-12)
    pd.DataFrame(kernel_rows).to_csv(OUT / "kernel_neighbors.csv", index=False)

    print("Running trajectory-cluster bootstrap...")
    transitions = fitted["transitions"]
    train_nojump = np.isin(
        transitions["trajectory"],
        fitted["train_ids"],
    ) & ~transitions["jump"]

    trajectory = transitions["trajectory"][train_nojump]
    X = transitions["X"][train_nojump]
    Y = transitions["dX"][train_nojump] / transitions["tau"]
    exposure = {}
    jump_count = {}
    for tid in np.unique(trajectory):
        exposure[int(tid)] = sum(
            integrated_exposure(
                baseline.dataframe,
                [int(tid)],
                name,
            )
            for name in ("Pexc0", "Pexc1")
        )
        jump_count[int(tid)] = len(baseline.collapse_channels[int(tid)])

    bootstrap = trajectory_cluster_bootstrap_rates(
        X,
        Y,
        trajectory,
        exposure,
        jump_count,
        reps=80,
        seed=36346,
    )
    pd.DataFrame(
        [
            {
                "g_point": bootstrap["g_point"],
                "g_CI95_low": bootstrap["g_CI95"][0],
                "g_CI95_high": bootstrap["g_CI95"][1],
                "gamma_point": bootstrap["gamma_point"],
                "gamma_CI95_low": bootstrap["gamma_CI95"][0],
                "gamma_CI95_high": bootstrap["gamma_CI95"][1],
            }
        ]
    ).to_csv(OUT / "cluster_bootstrap.csv", index=False)

    print(f"Robustness outputs: {OUT}")


if __name__ == "__main__":
    main()
