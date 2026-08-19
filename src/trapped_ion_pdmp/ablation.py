from __future__ import annotations

import numpy as np

from .km import kernel_drift_predict
from .pdmp import simulate_pdmp


def gaussian_local_moments(kernel_model, dX_train, X_query, tau: float):
    """Local drift and centered increment covariance for the Gaussian surrogate."""

    X_query = np.asarray(X_query, dtype=float)
    dX_train = np.asarray(dX_train, dtype=float)
    if len(X_query) == 0:
        dim = dX_train.shape[1]
        return {
            "drift": np.empty((0, dim)),
            "Q": np.empty((0, dim, dim)),
            "neighbor_radius": np.empty(0),
        }

    X_scaled = (X_query - kernel_model["mean"]) / kernel_model["scale"]
    distances, indices = kernel_model["nn"].kneighbors(X_scaled)
    h = np.maximum(np.median(distances, axis=1), 1e-12)
    weights = np.exp(-0.5 * (distances / h[:, None]) ** 2)
    sw = weights.sum(axis=1)

    neighbor_dx = dX_train[indices]
    mean_dx = np.einsum("nk,nkd->nd", weights, neighbor_dx) / sw[:, None]
    centered = neighbor_dx - mean_dx[:, None, :]
    covariance = np.einsum("nk,nki,nkj->nij", weights, centered, centered)
    covariance /= sw[:, None, None]

    Q = 0.5 * (covariance / tau + np.swapaxes(covariance / tau, 1, 2))
    return {
        "drift": mean_dx / tau,
        "Q": Q,
        "neighbor_radius": distances[:, -1],
    }


def simulate_gaussian_surrogate(
    kernel_model,
    dX_train,
    times,
    initial_state,
    tau: float,
    ntraj: int = 120,
    seed: int = 12345,
):
    """Moment-matched Gaussian surrogate with no clipping or projection."""

    times = np.asarray(times, dtype=float)
    rng = np.random.default_rng(seed)
    dim = len(initial_state)
    states = np.zeros((ntraj, len(times), dim), dtype=float)
    states[:, 0, :] = initial_state

    physical_samples = 0
    physical_violations = 0
    max_violation = 0.0
    numerical_failures = 0

    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        X = states[:, k, :]
        local = gaussian_local_moments(kernel_model, dX_train, X, tau)
        drift, Q = local["drift"], local["Q"]

        eigvals, eigvecs = np.linalg.eigh(Q)
        eigvals = np.maximum(eigvals, 0.0)
        B = eigvecs * np.sqrt(eigvals)[:, None, :]
        noise = np.einsum("nij,nj->ni", B, rng.normal(size=(ntraj, dim)))
        X_next = X + drift * dt + noise * np.sqrt(dt)

        finite = np.all(np.isfinite(X_next), axis=1)
        numerical_failures += int(np.sum(~finite))
        X_next[~finite] = X[~finite]

        p0, p1 = X_next[:, 0], X_next[:, 1]
        violation = (
            (p0 < -1e-6)
            | (p1 < -1e-6)
            | (p0 > 1.0 + 1e-6)
            | (p1 > 1.0 + 1e-6)
            | (p0 + p1 > 1.0 + 1e-6)
        )
        physical_samples += len(X_next)
        physical_violations += int(np.sum(violation))
        magnitude = np.maximum.reduce(
            [
                np.maximum(-p0, 0.0),
                np.maximum(-p1, 0.0),
                np.maximum(p0 - 1.0, 0.0),
                np.maximum(p1 - 1.0, 0.0),
                np.maximum(p0 + p1 - 1.0, 0.0),
            ]
        )
        max_violation = max(max_violation, float(np.max(magnitude)))
        states[:, k + 1, :] = X_next

    return {
        "states": states,
        "jump_times": np.full(ntraj, np.nan),
        "physical_violation_fraction": physical_violations / max(physical_samples, 1),
        "max_population_violation": max_violation,
        "numerical_failure_fraction": numerical_failures / max(physical_samples, 1),
    }


def simulate_drift_only(
    kernel_model,
    dX_train,
    drift_tau,
    times,
    initial_state,
    ntraj: int = 120,
    seed: int = 12345,
):
    return simulate_pdmp(
        kernel_model=kernel_model,
        dX_train=dX_train,
        drift_tau=drift_tau,
        times=times,
        initial_state=initial_state,
        gamma_hat=0.0,
        jump_target=np.zeros_like(initial_state, dtype=float),
        ntraj=ntraj,
        substeps=1,
        seed=seed,
    )
