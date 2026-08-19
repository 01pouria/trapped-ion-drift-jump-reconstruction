from dataclasses import dataclass, field

import numpy as np

GLOBAL_SEED = 12345
ACTIVE_STATE_COLUMNS = ("Pexc0", "Pexc1", "Csm0", "Csm1")


@dataclass
class TrappedIonConfig:
    """Configuration for N ions coupled to one collective phonon mode."""

    N: int = 2
    Nm: int = 4
    eta: float = 0.15
    nu: float = 1.0
    mode_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.N < 1:
            raise ValueError("N must be at least 1.")
        if self.Nm < 2:
            raise ValueError("Nm must be at least 2.")
        if self.eta < 0:
            raise ValueError("eta must be non-negative.")
        if self.nu <= 0:
            raise ValueError("nu must be positive.")

        if self.mode_weights is None:
            self.mode_weights = np.ones(self.N, dtype=float) / np.sqrt(self.N)
        else:
            weights = np.asarray(self.mode_weights, dtype=float)
            if weights.size != self.N:
                raise ValueError("mode_weights must have N entries.")
            norm = np.linalg.norm(weights)
            if norm == 0:
                raise ValueError("mode_weights cannot be zero.")
            self.mode_weights = weights / norm


@dataclass(frozen=True)
class BaselineExperiment:
    """Parameters used for the manuscript baseline."""

    eta: float = 0.15
    Omega: float = 1.0
    T1: float = 80.0
    Nm: int = 4
    ntraj: int = 200
    n_time: int = 1001
    window_periods: float = 2.2
    kernel_neighbors: int = 400
    train_fraction: float = 0.70
    active_tol: float = 1e-8
    seed: int = GLOBAL_SEED

    @property
    def transfer_time(self) -> float:
        return 2.0 * np.pi / (self.eta * self.Omega)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(
            0.0,
            self.window_periods * self.transfer_time,
            self.n_time,
        )

    @property
    def g_theory(self) -> float:
        return self.eta * self.Omega / (2.0 * np.sqrt(2.0))

    @property
    def gamma_theory(self) -> float:
        return 1.0 / self.T1


PARAMETRIC_CASES = (
    ("eta_0.10", 0.10, 1.00, 80.0),
    ("baseline", 0.15, 1.00, 80.0),
    ("eta_0.20", 0.20, 1.00, 80.0),
    ("Omega_0.70", 0.15, 0.70, 80.0),
    ("Omega_1.30", 0.15, 1.30, 80.0),
    ("T1_50", 0.15, 1.00, 50.0),
    ("T1_120", 0.15, 1.00, 120.0),
)
