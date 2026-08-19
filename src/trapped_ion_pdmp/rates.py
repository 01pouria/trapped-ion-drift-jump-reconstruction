from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ACTIVE_STATE_COLUMNS
from .km import (
    build_active_transitions,
    fit_polynomial_conditional_mean,
    make_trajectory_split,
    predict_polynomial_model,
)


def integrated_exposure(dataframe: pd.DataFrame, trajectory_ids, observable: str) -> float:
    """Trapezoidal exposure integral for a state-dependent event rate."""

    subset = dataframe[dataframe["trajectory"].isin(trajectory_ids)]
    total = 0.0

    for _, group in subset.groupby("trajectory", sort=False):
        group = group.sort_values("t")
        t = group["t"].to_numpy(dtype=float)
        x = group[observable].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        total += np.sum(0.5 * (x[:-1] + x[1:]) * np.diff(t))

    return float(total)


def collapse_count(collapse_channels: dict[int, np.ndarray], trajectory_ids, channel=None) -> int:
    count = 0
    for trajectory_id in trajectory_ids:
        channels = np.asarray(collapse_channels[int(trajectory_id)], dtype=int)
        count += len(channels) if channel is None else np.sum(channels == channel)
    return int(count)


def recover_coupling(
    dataframe: pd.DataFrame,
    jump_times: dict[int, np.ndarray],
    eta: float,
    Omega: float,
    mode_weights: np.ndarray,
    lag_steps: int = 1,
    seed: int = 12345,
    active_tol: float = 1e-8,
):
    train_ids, validation_ids = make_trajectory_split(
        dataframe, train_fraction=0.70, seed=seed
    )
    transitions = build_active_transitions(
        dataframe,
        ACTIVE_STATE_COLUMNS,
        lag_steps,
        jump_times,
        active_tol,
    )

    trajectory = transitions["trajectory"]
    train_mask = np.isin(trajectory, train_ids) & ~transitions["jump"]
    validation_mask = np.isin(trajectory, validation_ids) & ~transitions["jump"]

    X_train = transitions["X"][train_mask]
    Y_train = transitions["dX"][train_mask] / transitions["tau"]
    model = fit_polynomial_conditional_mean(
        X_train, Y_train, ACTIVE_STATE_COLUMNS, degree=1
    )

    i0 = model["feature_names"].index("Csm0")
    i1 = model["feature_names"].index("Csm1")
    g0_hat = float(model["coef_raw"][i0, 0])
    g1_hat = float(model["coef_raw"][i1, 1])
    g_hat = 0.5 * (g0_hat + g1_hat)

    g0_true = eta * Omega * mode_weights[0] / 2.0
    g1_true = eta * Omega * mode_weights[1] / 2.0
    g_true = 0.5 * (g0_true + g1_true)

    X_val = transitions["X"][validation_mask]
    Y_val = transitions["dX"][validation_mask] / transitions["tau"]
    Y_pred = predict_polynomial_model(model, X_val)
    drift_rmse = np.sqrt(np.mean((Y_pred - Y_val) ** 2))
    drift_scale = np.sqrt(np.mean(Y_val**2))
    drift_nrmse = drift_rmse / max(drift_scale, 1e-12)

    return {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "g0_hat": g0_hat,
        "g1_hat": g1_hat,
        "g_hat": float(g_hat),
        "g_true": float(g_true),
        "g_relative_error": float(abs(g_hat - g_true) / abs(g_true)),
        "drift_NRMSE": float(drift_nrmse),
        "model": model,
    }


def recover_gamma(
    dataframe: pd.DataFrame,
    collapse_channels: dict[int, np.ndarray],
    T1: float,
    train_ids,
    validation_ids,
):
    train_exposure = sum(
        integrated_exposure(dataframe, train_ids, name)
        for name in ("Pexc0", "Pexc1")
    )
    validation_exposure = sum(
        integrated_exposure(dataframe, validation_ids, name)
        for name in ("Pexc0", "Pexc1")
    )

    train_jumps = collapse_count(collapse_channels, train_ids)
    validation_jumps = collapse_count(collapse_channels, validation_ids)
    gamma_hat = train_jumps / train_exposure
    gamma_true = 1.0 / T1

    return {
        "gamma_true": gamma_true,
        "gamma_hat": float(gamma_hat),
        "gamma_relative_error": float(abs(gamma_hat - gamma_true) / gamma_true),
        "train_jumps": train_jumps,
        "validation_jumps": validation_jumps,
        "predicted_validation_jumps": float(gamma_hat * validation_exposure),
        "train_exposure": float(train_exposure),
        "validation_exposure": float(validation_exposure),
    }


def fit_zero_intercept_scaling(x, y):
    """Fit y = slope*x and report the usual coefficient of determination."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.dot(x, y) / np.dot(x, x))
    pred = slope * x
    residual = np.sum((y - pred) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return {"slope": slope, "R2": float(r2), "prediction": pred}
