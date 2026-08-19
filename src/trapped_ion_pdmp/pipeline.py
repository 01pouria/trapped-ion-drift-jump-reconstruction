from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ACTIVE_STATE_COLUMNS, BaselineExperiment, TrappedIonConfig
from .km import build_active_transitions, fit_kernel_neighbor_model, make_trajectory_split
from .metrics import evaluate_reconstruction
from .observables import build_baseline_observables, build_ensemble_dataframe
from .pdmp import simulate_pdmp
from .quantum import (
    TrappedIonHamiltonian,
    TrappedIonOperators,
    build_collapse_operators,
    computational_state,
    extract_jump_metadata,
    simulate_mc_ensemble,
)
from .rates import integrated_exposure


@dataclass
class BaselineData:
    config: BaselineExperiment
    runs: np.ndarray
    dataframe: pd.DataFrame
    times: np.ndarray
    observable_names: list[str]
    jump_times: dict[int, np.ndarray]
    collapse_channels: dict[int, np.ndarray]


def simulate_baseline(config: BaselineExperiment | None = None) -> BaselineData:
    """Generate the baseline quantum-jump ensemble used for inference."""

    config = config or BaselineExperiment()
    cfg = TrappedIonConfig(N=2, Nm=config.Nm, eta=config.eta)
    ops = TrappedIonOperators(cfg)
    ham = TrappedIonHamiltonian(cfg, ops)
    H = ham.red_sideband(Omega=config.Omega)
    psi0 = computational_state(cfg, [1, 0], 0)
    c_ops = build_collapse_operators(cfg, ops, T1=config.T1)

    observables = build_baseline_observables(cfg, ops)
    names = list(observables)
    result, runs = simulate_mc_ensemble(
        H,
        psi0,
        config.times,
        c_ops,
        [observables[name] for name in names],
        ntraj=config.ntraj,
        seed=config.seed,
    )
    jump_times, collapse_channels, jump_counts = extract_jump_metadata(result)
    dataframe = build_ensemble_dataframe(runs, config.times, names, jump_counts)

    return BaselineData(
        config=config,
        runs=runs,
        dataframe=dataframe,
        times=config.times,
        observable_names=names,
        jump_times=jump_times,
        collapse_channels=collapse_channels,
    )


def fit_pdmp_components(data: BaselineData):
    """Fit drift data, the relaxation rate, and the empirical jump target."""

    cfg = data.config
    train_ids, validation_ids = make_trajectory_split(
        data.dataframe,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )
    transitions = build_active_transitions(
        data.dataframe,
        ACTIVE_STATE_COLUMNS,
        lag_steps=1,
        jump_times_by_trajectory=data.jump_times,
        active_tol=cfg.active_tol,
    )

    train = np.isin(transitions["trajectory"], train_ids)
    nojump = train & ~transitions["jump"]
    jump = train & transitions["jump"]

    X_train = transitions["X"][nojump]
    dX_train = transitions["dX"][nojump]
    kernel = fit_kernel_neighbor_model(X_train, cfg.kernel_neighbors)

    exposure = sum(
        integrated_exposure(data.dataframe, train_ids, name)
        for name in ("Pexc0", "Pexc1")
    )
    jump_count = sum(len(data.collapse_channels[int(tid)]) for tid in train_ids)
    gamma_hat = jump_count / exposure

    X_before = transitions["X"][jump]
    X_after = X_before + transitions["dX"][jump]
    jump_target = X_after.mean(axis=0)

    name_to_idx = {name: i for i, name in enumerate(data.observable_names)}
    quantum_validation = np.stack(
        [data.runs[name_to_idx[name], validation_ids, :] for name in ACTIVE_STATE_COLUMNS],
        axis=-1,
    )

    first_jump = np.full(len(validation_ids), np.nan)
    for i, tid in enumerate(validation_ids):
        jt = data.jump_times[int(tid)]
        if len(jt):
            first_jump[i] = jt[0]

    return {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "transitions": transitions,
        "kernel_model": kernel,
        "X_drift_train": X_train,
        "dX_drift_train": dX_train,
        "gamma_hat": float(gamma_hat),
        "jump_target": jump_target,
        "quantum_validation": quantum_validation,
        "quantum_first_jump": first_jump,
    }



def simulate_parametric_case(
    name: str,
    eta: float,
    Omega: float,
    T1: float,
    ntraj: int = 120,
    n_time: int = 801,
    seed: int = 1000,
):
    """Generate one parameter-sweep ensemble."""

    cfg = TrappedIonConfig(N=2, Nm=4, eta=eta)
    ops = TrappedIonOperators(cfg)
    ham = TrappedIonHamiltonian(cfg, ops)
    H = ham.red_sideband(Omega=Omega)
    psi0 = computational_state(cfg, [1, 0], 0)
    c_ops = build_collapse_operators(cfg, ops, T1=T1)

    transfer_time = 2.0 * np.pi / (eta * Omega)
    times = np.linspace(0.0, 2.2 * transfer_time, n_time)

    from .observables import build_parametric_observables

    observables = build_parametric_observables(cfg, ops)
    names = list(observables)
    result, runs = simulate_mc_ensemble(
        H,
        psi0,
        times,
        c_ops,
        [observables[name] for name in names],
        ntraj=ntraj,
        seed=seed,
    )
    jump_times, collapse_channels, jump_counts = extract_jump_metadata(result)
    dataframe = build_ensemble_dataframe(runs, times, names, jump_counts)

    return {
        "name": name,
        "eta": eta,
        "Omega": Omega,
        "T1": T1,
        "cfg": cfg,
        "dataframe": dataframe,
        "jump_times": jump_times,
        "collapse_channels": collapse_channels,
        "times": times,
        "transfer_time": transfer_time,
    }


def reconstruct_and_evaluate(
    data: BaselineData,
    fitted,
    ntraj: int = 300,
    substeps: int = 1,
    seed: int | None = None,
):
    """Run the independent reconstructed PDMP and evaluate held-out statistics."""

    seed = data.config.seed + 30001 if seed is None else seed
    initial_state = np.array([1.0, 0.0, 0.0, 0.0])
    reconstruction = simulate_pdmp(
        fitted["kernel_model"],
        fitted["dX_drift_train"],
        fitted["transitions"]["tau"],
        data.times,
        initial_state,
        fitted["gamma_hat"],
        fitted["jump_target"],
        ntraj=ntraj,
        substeps=substeps,
        seed=seed,
    )

    checkpoint_indices = [
        min(int(round(f * (len(data.times) - 1))), len(data.times) - 1)
        for f in (0.25, 0.50, 0.75, 1.00)
    ]
    evaluation = evaluate_reconstruction(
        reconstruction,
        fitted["quantum_validation"],
        fitted["quantum_first_jump"],
        data.times,
        ACTIVE_STATE_COLUMNS,
        checkpoint_indices,
    )
    return reconstruction, evaluation
