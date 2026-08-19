from __future__ import annotations

import numpy as np
import pandas as pd


def build_spin_motion_observables(cfg, ops):
    """Spin-motion coherences used in the reduced state."""

    extra = {}
    for i in range(cfg.N):
        A = ops.a * ops.Sp(i)
        A_dag = ops.adag * ops.Sm(i)
        extra[f"Csm{i}"] = A + A_dag
        extra[f"Jsm{i}"] = 1j * (A - A_dag)

    if cfg.N >= 2:
        exchange = ops.Sp(0) * ops.Sm(1)
        exchange_dag = ops.Sm(0) * ops.Sp(1)
        extra["C01"] = exchange + exchange_dag
        extra["J01"] = -1j * (exchange - exchange_dag)

    for name, op in extra.items():
        if not op.isherm:
            raise RuntimeError(f"{name} is not Hermitian.")
    return extra


def build_baseline_observables(cfg, ops):
    spin_motion = build_spin_motion_observables(cfg, ops)
    base = {
        "X0": ops.X(0),
        "Y0": ops.Y(0),
        "Z0": ops.Z(0),
        "X1": ops.X(1),
        "Y1": ops.Y(1),
        "Z1": ops.Z(1),
        "ZZ": ops.Z(0) * ops.Z(1),
        "Pexc0": ops.P1(0),
        "Pexc1": ops.P1(1),
        "n_ph": ops.n,
        "N_exc": ops.P1(0) + ops.P1(1) + ops.n,
    }
    return {**base, **spin_motion}


def build_parametric_observables(cfg, ops):
    spin_motion = build_spin_motion_observables(cfg, ops)
    return {
        "Pexc0": ops.P1(0),
        "Pexc1": ops.P1(1),
        "n_ph": ops.n,
        "N_exc": ops.P1(0) + ops.P1(1) + ops.n,
        "Csm0": spin_motion["Csm0"],
        "Csm1": spin_motion["Csm1"],
    }


def build_ensemble_dataframe(
    runs: np.ndarray,
    times: np.ndarray,
    observable_names: list[str],
    jump_counts: np.ndarray | None = None,
) -> pd.DataFrame:
    n_obs, n_traj, n_times = runs.shape
    if n_obs != len(observable_names):
        raise ValueError("Observable names do not match the trajectory array.")

    data = {
        "trajectory": np.repeat(np.arange(n_traj), n_times),
        "t": np.tile(times, n_traj),
    }
    for i, name in enumerate(observable_names):
        data[name] = runs[i].reshape(-1)

    if {"ZZ", "Z0", "Z1"}.issubset(data):
        data["Czz"] = data["ZZ"] - data["Z0"] * data["Z1"]

    if jump_counts is not None:
        data["trajectory_total_jumps"] = np.repeat(jump_counts, n_times)

    return pd.DataFrame(data)
