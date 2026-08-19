from __future__ import annotations

import numpy as np

from .km import kernel_drift_predict


def simulate_pdmp(
    kernel_model,
    dX_train,
    drift_tau: float,
    times,
    initial_state,
    gamma_hat: float,
    jump_target,
    ntraj: int = 300,
    substeps: int = 2,
    seed: int = 12345,
):
    """Simulate the inferred drift--jump model with midpoint integration."""

    times = np.asarray(times, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    jump_target = np.asarray(jump_target, dtype=float)
    rng = np.random.default_rng(seed)

    states = np.zeros((ntraj, len(times), len(initial_state)), dtype=float)
    states[:, 0, :] = initial_state
    alive = np.ones(ntraj, dtype=bool)
    jump_times = np.full(ntraj, np.nan)
    support_radii = []

    physical_samples = 0
    physical_violations = 0
    max_violation = 0.0

    for k in range(len(times) - 1):
        h = (times[k + 1] - times[k]) / substeps
        working = states[:, k, :].copy()

        for s in range(substeps):
            active_idx = np.flatnonzero(alive)
            if active_idx.size == 0:
                break

            X = working[active_idx]
            k1 = kernel_drift_predict(kernel_model, dX_train, X, drift_tau)["D1"]
            X_mid = X + 0.5 * h * k1
            stage2 = kernel_drift_predict(kernel_model, dX_train, X_mid, drift_tau)
            X_flow = X + h * stage2["D1"]
            support_radii.append(stage2["neighbor_radius"])

            p0, p1 = X_flow[:, 0], X_flow[:, 1]
            violation = (
                (p0 < -1e-6)
                | (p1 < -1e-6)
                | (p0 > 1.0 + 1e-6)
                | (p1 > 1.0 + 1e-6)
                | (p0 + p1 > 1.0 + 1e-6)
            )
            physical_samples += len(X_flow)
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

            # The hazard uses the physical excited-population range.
            excited_mid = np.clip(X_mid[:, 0] + X_mid[:, 1], 0.0, 1.0)
            hazard = gamma_hat * excited_mid
            jump_probability = 1.0 - np.exp(-hazard * h)
            jump_now = rng.random(len(active_idx)) < jump_probability

            next_active = X_flow.copy()
            if np.any(jump_now):
                next_active[jump_now] = jump_target
                jumped = active_idx[jump_now]
                alive[jumped] = False
                jump_times[jumped] = times[k] + (s + 1) * h

            working[active_idx] = next_active

        states[:, k + 1, :] = working

    radii = np.concatenate(support_radii) if support_radii else np.array([])
    return {
        "states": states,
        "jump_times": jump_times,
        "support_radii": radii,
        "physical_violation_fraction": physical_violations / max(physical_samples, 1),
        "max_population_violation": max_violation,
        "substeps": substeps,
    }
