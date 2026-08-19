from __future__ import annotations

import numpy as np
import pandas as pd

from .km import build_active_transitions, fit_kernel_neighbor_model, kernel_km_predict

from .config import TrappedIonConfig
from .quantum import (
    TrappedIonHamiltonian,
    TrappedIonOperators,
    build_collapse_operators,
    computational_state,
    plus_state,
    simulate_master,
)


def simulator_validation():
    """Run the compact analytic checks reported in the appendix."""

    rows = []

    cfg = TrappedIonConfig(N=1, Nm=4, eta=0.12)
    ops = TrappedIonOperators(cfg)
    ham = TrappedIonHamiltonian(cfg, ops)

    Omega = 1.0
    t = np.linspace(0, 4 * np.pi / Omega, 1001)
    result = simulate_master(
        ham.carrier(Omega=Omega),
        computational_state(cfg, [0], 0),
        t,
        {"P_exc": ops.P1(0)},
    )
    exact = np.sin(Omega * t / 2) ** 2
    rows.append(("Carrier dynamics", np.max(np.abs(result["P_exc"] - exact))))

    t = np.linspace(0, 2 * np.pi / (cfg.eta * Omega), 1001)
    exact = np.sin(cfg.eta * Omega * t / 2) ** 2
    result = simulate_master(
        ham.red_sideband(Omega=Omega),
        computational_state(cfg, [0], 1),
        t,
        {"P_exc": ops.P1(0)},
    )
    rows.append(("Red-sideband dynamics", np.max(np.abs(result["P_exc"] - exact))))

    result = simulate_master(
        ham.blue_sideband(Omega=Omega),
        computational_state(cfg, [0], 0),
        t,
        {"P_exc": ops.P1(0)},
    )
    rows.append(("Blue-sideband dynamics", np.max(np.abs(result["P_exc"] - exact))))

    cfg2 = TrappedIonConfig(N=2, Nm=4, eta=0.15)
    ops2 = TrappedIonOperators(cfg2)
    ham2 = TrappedIonHamiltonian(cfg2, ops2)
    transfer = 2 * np.pi / cfg2.eta
    t2 = np.linspace(0, 2.2 * transfer, 1601)
    N_exc = ops2.P1(0) + ops2.P1(1) + ops2.n
    result = simulate_master(
        ham2.red_sideband(Omega=1.0),
        computational_state(cfg2, [1, 0], 0),
        t2,
        {"N_exc": N_exc},
    )
    rows.append(
        (
            "Two-ion excitation conservation",
            np.max(np.abs(result["N_exc"] - result["N_exc"][0])),
        )
    )

    cfg1 = TrappedIonConfig(N=1, Nm=3, eta=0.10)
    ops1 = TrappedIonOperators(cfg1)
    H0 = 0.0 * ops1.Z(0)
    T1 = 8.0
    t1 = np.linspace(0, 5 * T1, 601)
    result = simulate_master(
        H0,
        computational_state(cfg1, [1], 0),
        t1,
        {"P_exc": ops1.P1(0)},
        build_collapse_operators(cfg1, ops1, T1=T1),
    )
    rows.append(("T1 relaxation", np.max(np.abs(result["P_exc"] - np.exp(-t1 / T1)))))

    return pd.DataFrame(rows, columns=["test", "absolute_error"])


def nojump_path_metrics(quantum_path, reconstructed_path, state_columns):
    """Direct path metrics for trajectories conditioned on no jump."""

    quantum_path = np.asarray(quantum_path, dtype=float)
    reconstructed_path = np.asarray(reconstructed_path, dtype=float)
    rows = []

    for j, name in enumerate(state_columns):
        diff = reconstructed_path[:, j] - quantum_path[:, j]
        rmse = np.sqrt(np.mean(diff**2))
        scale = max(np.ptp(quantum_path[:, j]), 1e-12)
        rows.append(
            {
                "variable": name,
                "RMSE": rmse,
                "NRMSE": rmse / scale,
                "max_abs_error": np.max(np.abs(diff)),
                "mean_bias": np.mean(diff),
                "correlation": np.corrcoef(
                    quantum_path[:, j], reconstructed_path[:, j]
                )[0, 1],
            }
        )

    return pd.DataFrame(rows)


def finite_lag_km_summary(
    dataframe,
    jump_times,
    train_ids,
    validation_ids,
    state_columns,
    lags=(1, 2, 4, 8),
    n_neighbors: int = 400,
    active_tol: float = 1e-8,
    seed: int = 12345,
    max_probe: int = 1000,
):
    """Summarize drift error and second-moment scaling across finite lags."""

    rows = []
    for lag_steps in lags:
        transitions = build_active_transitions(
            dataframe,
            state_columns,
            lag_steps,
            jump_times,
            active_tol,
        )
        train_mask = np.isin(transitions["trajectory"], train_ids)
        val_mask = np.isin(transitions["trajectory"], validation_ids)
        train_cont = train_mask & ~transitions["jump"]
        val_cont = val_mask & ~transitions["jump"]

        Xtr = transitions["X"][train_cont]
        dXtr = transitions["dX"][train_cont]
        Xva = transitions["X"][val_cont]
        dXva = transitions["dX"][val_cont]

        model = fit_kernel_neighbor_model(Xtr, n_neighbors)
        rng = np.random.default_rng(seed + lag_steps)
        nq = min(max_probe, len(Xva))
        idx = rng.choice(len(Xva), size=nq, replace=False)
        pred = kernel_km_predict(
            model,
            dXtr,
            Xva[idx],
            transitions["tau"],
        )

        actual = dXva[idx] / transitions["tau"]
        drift_error = np.sqrt(np.mean((pred["D1"] - actual) ** 2))
        drift_scale = np.sqrt(np.mean(actual**2))
        eigs = np.linalg.eigvalsh(pred["D2"])
        traces = np.trace(pred["D2"], axis1=1, axis2=2)

        rows.append(
            {
                "lag_steps": lag_steps,
                "tau": transitions["tau"],
                "n_cont_train": int(np.sum(train_cont)),
                "n_jump_windows": int(np.sum(transitions["jump"])),
                "drift_NRMSE": drift_error / max(drift_scale, 1e-12),
                "median_D2_trace": float(np.median(traces)),
                "median_D2_trace_over_tau": float(
                    np.median(traces) / transitions["tau"]
                ),
                "worst_D2_eigenvalue": float(np.min(eigs)),
                "PSD_violation_fraction": float(
                    np.mean(eigs[:, 0] < -1e-12)
                ),
            }
        )

    return pd.DataFrame(rows)
