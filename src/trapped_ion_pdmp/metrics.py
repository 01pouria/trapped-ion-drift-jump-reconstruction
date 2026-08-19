from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def survival_curve(jump_times, times):
    jump_times = np.asarray(jump_times, dtype=float)
    times = np.asarray(times, dtype=float)
    return np.asarray(
        [np.mean(~np.isfinite(jump_times) | (jump_times > t)) for t in times],
        dtype=float,
    )


def diagnose_state_domain(states):
    X = np.asarray(states, dtype=float).reshape(-1, states.shape[-1])
    p0, p1 = X[:, 0], X[:, 1]
    violation = (
        (p0 < -1e-6)
        | (p1 < -1e-6)
        | (p0 > 1.0 + 1e-6)
        | (p1 > 1.0 + 1e-6)
        | (p0 + p1 > 1.0 + 1e-6)
    )
    magnitude = np.maximum.reduce(
        [
            np.maximum(-p0, 0.0),
            np.maximum(-p1, 0.0),
            np.maximum(p0 - 1.0, 0.0),
            np.maximum(p1 - 1.0, 0.0),
            np.maximum(p0 + p1 - 1.0, 0.0),
        ]
    )
    return {
        "fraction": float(np.mean(violation)),
        "max_violation": float(np.max(magnitude)),
    }


def evaluate_reconstruction(
    reconstruction,
    quantum_validation,
    quantum_first_jump,
    times,
    state_columns,
    checkpoint_indices,
):
    """Evaluate all models with the same held-out metrics."""

    pdmp_states = np.asarray(reconstruction["states"], dtype=float)
    q = np.asarray(quantum_validation, dtype=float)
    times = np.asarray(times, dtype=float)

    q_mean, p_mean = q.mean(axis=0), pdmp_states.mean(axis=0)
    q_var = q.var(axis=0, ddof=1)
    p_var = pdmp_states.var(axis=0, ddof=1)

    moment_rows = []
    for j, name in enumerate(state_columns):
        mean_rmse = np.sqrt(np.mean((p_mean[:, j] - q_mean[:, j]) ** 2))
        var_rmse = np.sqrt(np.mean((p_var[:, j] - q_var[:, j]) ** 2))
        moment_rows.append(
            {
                "variable": name,
                "mean_NRMSE": mean_rmse / max(np.ptp(q_mean[:, j]), 1e-12),
                "variance_NRMSE": var_rmse / max(np.max(q_var[:, j]), 1e-12),
            }
        )
    moments = pd.DataFrame(moment_rows)

    wd_rows = []
    for idx in checkpoint_indices:
        for j, name in enumerate(state_columns):
            wd = wasserstein_distance(q[:, idx, j], pdmp_states[:, idx, j])
            scale = np.ptp(q[:, :, j])
            wd_rows.append(
                {
                    "variable": name,
                    "time": times[idx],
                    "normalized_wasserstein": wd / max(scale, 1e-12),
                }
            )
    wd_table = pd.DataFrame(wd_rows)

    q_survival = survival_curve(quantum_first_jump, times)
    p_survival = survival_curve(reconstruction["jump_times"], times)
    survival_rmse = np.sqrt(np.mean((p_survival - q_survival) ** 2))

    q_jump_fraction = np.mean(np.isfinite(quantum_first_jump))
    p_jump_fraction = np.mean(np.isfinite(reconstruction["jump_times"]))

    covariance_errors = []
    for idx in checkpoint_indices:
        q_cov = np.cov(q[:, idx, :], rowvar=False)
        p_cov = np.cov(pdmp_states[:, idx, :], rowvar=False)
        covariance_errors.append(
            np.linalg.norm(p_cov - q_cov, ord="fro")
            / max(np.linalg.norm(q_cov, ord="fro"), 1e-12)
        )

    return {
        "summary": {
            "mean_NRMSE": float(moments["mean_NRMSE"].mean()),
            "variance_NRMSE": float(moments["variance_NRMSE"].mean()),
            "wasserstein": float(wd_table["normalized_wasserstein"].mean()),
            "survival_RMSE": float(survival_rmse),
            "jump_fraction_error": float(abs(q_jump_fraction - p_jump_fraction)),
            "mean_covariance_error": float(np.mean(covariance_errors)),
            "physical_violation_fraction": float(
                reconstruction.get("physical_violation_fraction", np.nan)
            ),
            "max_population_violation": float(
                reconstruction.get("max_population_violation", np.nan)
            ),
        },
        "moments": moments,
        "wasserstein": wd_table,
        "quantum_survival": q_survival,
        "model_survival": p_survival,
    }
