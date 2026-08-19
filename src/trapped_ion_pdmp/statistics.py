from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression


def poisson_rate_ci(events: int, exposure: float, alpha: float = 0.05):
    """Exact confidence interval for a Poisson event rate."""

    events = int(events)
    exposure = float(exposure)
    if exposure <= 0:
        return np.nan, np.nan

    lower = 0.0
    if events > 0:
        lower = 0.5 * chi2.ppf(alpha / 2, 2 * events) / exposure
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / exposure
    return float(lower), float(upper)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """Wilson interval for a binomial proportion."""

    successes, total = int(successes), int(total)
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def trajectory_cluster_bootstrap_rates(
    X,
    Y,
    trajectory_ids,
    exposure_by_trajectory,
    jump_count_by_trajectory,
    reps: int = 80,
    seed: int = 36346,
):
    """Cluster-bootstrap g and gamma by resampling whole trajectories."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    trajectory_ids = np.asarray(trajectory_ids, dtype=int)

    unique_ids = np.sort(np.unique(trajectory_ids))
    row_position = np.searchsorted(unique_ids, trajectory_ids)

    exposure = np.asarray(
        [exposure_by_trajectory[int(tid)] for tid in unique_ids],
        dtype=float,
    )
    jumps = np.asarray(
        [jump_count_by_trajectory[int(tid)] for tid in unique_ids],
        dtype=float,
    )

    point_model = LinearRegression(fit_intercept=True).fit(X, Y)
    g_point = 0.5 * (point_model.coef_[0, 2] + point_model.coef_[1, 3])
    gamma_point = jumps.sum() / exposure.sum()

    rng = np.random.default_rng(seed)
    boot_g = np.empty(reps)
    boot_gamma = np.empty(reps)
    n_clusters = len(unique_ids)

    for b in range(reps):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled, minlength=n_clusters)
        row_weights = multiplicity[row_position]

        model = LinearRegression(fit_intercept=True)
        model.fit(X, Y, sample_weight=row_weights)
        boot_g[b] = 0.5 * (model.coef_[0, 2] + model.coef_[1, 3])

        exposure_b = exposure[sampled].sum()
        jumps_b = jumps[sampled].sum()
        boot_gamma[b] = jumps_b / exposure_b

    return {
        "g_point": float(g_point),
        "gamma_point": float(gamma_point),
        "g_samples": boot_g,
        "gamma_samples": boot_gamma,
        "g_CI95": tuple(np.percentile(boot_g, [2.5, 97.5])),
        "gamma_CI95": tuple(np.percentile(boot_gamma, [2.5, 97.5])),
    }
