from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors


def build_active_transitions(
    dataframe: pd.DataFrame,
    state_columns,
    lag_steps: int,
    jump_times_by_trajectory: dict[int, np.ndarray],
    active_tol: float = 1e-8,
):
    """Build active-origin transitions without crossing trajectory boundaries."""

    X_blocks, dX_blocks = [], []
    trajectory_blocks, t0_blocks, t1_blocks, jump_blocks = [], [], [], []

    for trajectory_id, group in dataframe.groupby("trajectory", sort=False):
        group = group.sort_values("t").reset_index(drop=True)
        if len(group) <= lag_steps:
            continue

        t = group["t"].to_numpy(dtype=float)
        X = group[list(state_columns)].to_numpy(dtype=float)
        activity = group["N_exc"].to_numpy(dtype=float)

        X0, X1 = X[:-lag_steps], X[lag_steps:]
        t0, t1 = t[:-lag_steps], t[lag_steps:]
        active = activity[:-lag_steps] > active_tol

        jumps = np.asarray(
            jump_times_by_trajectory.get(int(trajectory_id), []),
            dtype=float,
        )
        jump_within = np.zeros(len(X0), dtype=bool)
        if jumps.size:
            left = np.searchsorted(jumps, t0, side="right")
            right = np.searchsorted(jumps, t1, side="right")
            jump_within = right > left

        X_blocks.append(X0[active])
        dX_blocks.append((X1 - X0)[active])
        trajectory_blocks.append(np.full(np.sum(active), trajectory_id, dtype=int))
        t0_blocks.append(t0[active])
        t1_blocks.append(t1[active])
        jump_blocks.append(jump_within[active])

    if not X_blocks:
        raise ValueError("No active transitions were found.")

    X = np.vstack(X_blocks)
    dX = np.vstack(dX_blocks)
    trajectory = np.concatenate(trajectory_blocks)
    t0 = np.concatenate(t0_blocks)
    t1 = np.concatenate(t1_blocks)
    jump = np.concatenate(jump_blocks)

    tau_values = t1 - t0
    tau = float(np.median(tau_values))
    if np.max(np.abs(tau_values - tau)) > 1e-10:
        raise RuntimeError("Non-uniform lag detected.")

    return {
        "X": X,
        "dX": dX,
        "trajectory": trajectory,
        "t0": t0,
        "t1": t1,
        "tau": tau,
        "jump": jump,
    }


def make_trajectory_split(
    dataframe: pd.DataFrame,
    train_fraction: float = 0.70,
    seed: int = 12345,
):
    """Split whole trajectories, stratifying by jump occurrence when possible."""

    meta = dataframe.groupby("trajectory").first()
    ids = meta.index.to_numpy()
    stratify = None

    if "trajectory_total_jumps" in meta.columns:
        labels = (meta["trajectory_total_jumps"].to_numpy() > 0).astype(int)
        if np.unique(labels).size > 1:
            stratify = labels

    train_ids, validation_ids = train_test_split(
        ids,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(train_ids), np.sort(validation_ids)


def fit_kernel_neighbor_model(X_train: np.ndarray, n_neighbors: int = 400):
    X_train = np.asarray(X_train, dtype=float)
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    X_scaled = (X_train - mean) / scale

    k = min(n_neighbors, len(X_train))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(X_scaled)
    return {"X_train": X_train, "mean": mean, "scale": scale, "nn": nn, "k": k}


def _kernel_weights(distances: np.ndarray) -> np.ndarray:
    h = np.median(distances, axis=-1)
    h = np.maximum(h, 1e-12)
    return np.exp(-0.5 * (distances / np.expand_dims(h, -1)) ** 2)


def kernel_km_predict(kernel_model, dX_train, X_query, tau: float):
    """Estimate finite-lag first and second KM coefficients."""

    X_query = np.asarray(X_query, dtype=float)
    dX_train = np.asarray(dX_train, dtype=float)
    X_scaled = (X_query - kernel_model["mean"]) / kernel_model["scale"]
    distances, indices = kernel_model["nn"].kneighbors(X_scaled)
    weights = _kernel_weights(distances)
    sw = weights.sum(axis=1)

    neighbor_dx = dX_train[indices]
    mean_dx = np.einsum("nk,nkd->nd", weights, neighbor_dx) / sw[:, None]
    second = np.einsum("nk,nki,nkj->nij", weights, neighbor_dx, neighbor_dx)
    second /= sw[:, None, None]

    effective = sw**2 / np.sum(weights**2, axis=1)
    return {
        "D1": mean_dx / tau,
        "D2": second / (2.0 * tau),
        "effective_samples": effective,
        "neighbor_radius": distances[:, -1],
    }


def kernel_drift_predict(kernel_model, dX_train, X_query, tau: float):
    """Fast drift-only query used by the reconstructed flow."""

    X_query = np.asarray(X_query, dtype=float)
    dX_train = np.asarray(dX_train, dtype=float)
    if len(X_query) == 0:
        return {
            "D1": np.empty((0, dX_train.shape[1])),
            "neighbor_radius": np.empty(0),
            "effective_samples": np.empty(0),
        }

    # Identical states share one nearest-neighbor query.
    unique, inverse = np.unique(X_query, axis=0, return_inverse=True)
    X_scaled = (unique - kernel_model["mean"]) / kernel_model["scale"]
    distances, indices = kernel_model["nn"].kneighbors(X_scaled)
    weights = _kernel_weights(distances)
    sw = weights.sum(axis=1)
    neighbor_dx = dX_train[indices]
    mean_dx = np.einsum("nk,nkd->nd", weights, neighbor_dx) / sw[:, None]

    return {
        "D1": (mean_dx / tau)[inverse],
        "neighbor_radius": distances[:, -1][inverse],
        "effective_samples": (sw**2 / np.sum(weights**2, axis=1))[inverse],
    }


def build_monomial_specs(dimension: int, degree: int):
    specs = [()]
    for p in range(1, degree + 1):
        specs.extend(combinations_with_replacement(range(dimension), p))
    return list(specs)


def build_polynomial_design(X: np.ndarray, specs):
    X = np.asarray(X, dtype=float)
    Phi = np.ones((X.shape[0], len(specs)), dtype=float)
    for j, spec in enumerate(specs):
        if spec:
            Phi[:, j] = np.prod(X[:, spec], axis=1)
    return Phi


def fit_polynomial_conditional_mean(
    X: np.ndarray,
    Y: np.ndarray,
    state_names,
    degree: int = 1,
    rcond: float = 1e-10,
):
    """Fit a scaled polynomial conditional mean with SVD least squares."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]

    specs = build_monomial_specs(X.shape[1], degree)
    feature_names = [
        "1" if not spec else "*".join(state_names[i] for i in spec)
        for spec in specs
    ]
    Phi = build_polynomial_design(X, specs)

    feature_mean = Phi.mean(axis=0)
    feature_scale = Phi.std(axis=0)
    feature_mean[0] = 0.0
    feature_scale[0] = 1.0
    feature_scale[feature_scale < 1e-14] = 1.0
    Phi_scaled = (Phi - feature_mean) / feature_scale

    coef_scaled, _, rank, singular_values = np.linalg.lstsq(
        Phi_scaled, Y, rcond=rcond
    )
    coef_raw = coef_scaled / feature_scale[:, None]
    coef_raw[0, :] = coef_scaled[0, :] - np.sum(
        (feature_mean[1:] / feature_scale[1:])[:, None] * coef_scaled[1:, :],
        axis=0,
    )

    condition_number = np.inf
    if singular_values.size:
        condition_number = singular_values[0] / max(singular_values[-1], 1e-300)

    return {
        "state_names": list(state_names),
        "degree": degree,
        "specs": specs,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coef_scaled": coef_scaled,
        "coef_raw": coef_raw,
        "rank": int(rank),
        "singular_values": singular_values,
        "condition_number": float(condition_number),
    }


def predict_polynomial_model(model, X: np.ndarray):
    Phi = build_polynomial_design(X, model["specs"])
    Phi_scaled = (Phi - model["feature_mean"]) / model["feature_scale"]
    return Phi_scaled @ model["coef_scaled"]


def state_space_matrix_diagnostics(
    dataframe: pd.DataFrame,
    columns,
    max_rows: int = 100000,
    seed: int = 12345,
    tol: float = 1e-10,
):
    X = dataframe[list(columns)].to_numpy(dtype=float)
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), size=max_rows, replace=False)]

    stds = X.std(axis=0)
    keep = stds > 1e-12
    kept = [c for c, k in zip(columns, keep) if k]
    removed = [c for c, k in zip(columns, keep) if not k]
    X = X[:, keep]

    if X.shape[1] == 0:
        return {
            "rank": 0,
            "dimension": 0,
            "condition_number": np.inf,
            "effective_dimension": 0.0,
            "kept": [],
            "removed": removed,
            "singular_values": np.array([]),
        }

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    rank = int(np.sum(s > tol * s[0]))
    condition = s[0] / s[-1] if s[-1] > 0 else np.inf
    weights = s**2
    effective_dimension = weights.sum() ** 2 / np.sum(weights**2)

    return {
        "rank": rank,
        "dimension": X.shape[1],
        "condition_number": float(condition),
        "effective_dimension": float(effective_dimension),
        "kept": kept,
        "removed": removed,
        "singular_values": s,
    }


def build_markov_dataset(dataframe: pd.DataFrame, columns, lag_steps: int = 4):
    past, now, future, groups = [], [], [], []

    for trajectory_id, group in dataframe.groupby("trajectory", sort=False):
        group = group.sort_values("t")
        X = group[list(columns)].to_numpy(dtype=float)
        if len(X) <= 2 * lag_steps:
            continue
        past.append(X[:-2 * lag_steps])
        now.append(X[lag_steps:-lag_steps])
        future.append(X[2 * lag_steps:])
        groups.append(np.full(len(X) - 2 * lag_steps, trajectory_id, dtype=int))

    return (
        np.vstack(past),
        np.vstack(now),
        np.vstack(future),
        np.concatenate(groups),
    )


def markov_memory_diagnostic(
    dataframe: pd.DataFrame,
    columns,
    lag_steps: int = 4,
    max_samples: int = 50000,
    seed: int = 12345,
):
    """Compare present-only and present-plus-past prediction."""

    Xpast, Xnow, Xfuture, groups = build_markov_dataset(
        dataframe, columns, lag_steps
    )

    if len(Xnow) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xnow), size=max_samples, replace=False)
        Xpast, Xnow, Xfuture, groups = (
            Xpast[idx],
            Xnow[idx],
            Xfuture[idx],
            groups[idx],
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(Xnow, groups=groups))

    present_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=seed,
    )
    memory_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=seed,
    )

    present_model.fit(Xnow[train_idx], Xfuture[train_idx])
    memory_features = np.hstack([Xnow, Xpast])
    memory_model.fit(memory_features[train_idx], Xfuture[train_idx])

    pred_present = present_model.predict(Xnow[test_idx])
    pred_memory = memory_model.predict(memory_features[test_idx])

    mse_present = mean_squared_error(
        Xfuture[test_idx], pred_present, multioutput="raw_values"
    )
    mse_memory = mean_squared_error(
        Xfuture[test_idx], pred_memory, multioutput="raw_values"
    )
    variance = np.maximum(np.var(Xfuture[test_idx], axis=0), 1e-12)
    nrmse_present = np.sqrt(mse_present / variance)
    nrmse_memory = np.sqrt(mse_memory / variance)

    avg_present = float(np.mean(nrmse_present))
    avg_memory = float(np.mean(nrmse_memory))
    gain = (avg_present - avg_memory) / max(avg_present, 1e-12)

    return {
        "average_NRMSE_present": avg_present,
        "average_NRMSE_memory": avg_memory,
        "memory_gain": float(gain),
        "NRMSE_present": nrmse_present,
        "NRMSE_memory": nrmse_memory,
    }
