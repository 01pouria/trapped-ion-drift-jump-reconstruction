from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .config import TrappedIonConfig


def _qutip():
    try:
        import qutip as qt
    except ImportError as exc:
        raise ImportError(
            "QuTiP is required for microscopic simulations. "
            "Install the project dependencies with `pip install -e .`."
        ) from exc
    return qt


class TrappedIonOperators:
    """Operator factory for ions followed by one phonon mode."""

    def __init__(self, cfg: TrappedIonConfig):
        qt = _qutip()
        self.cfg = cfg
        self.N = cfg.N
        self.Nm = cfg.Nm

        ket0 = qt.basis(2, 0)
        ket1 = qt.basis(2, 1)
        self.local_P0 = ket0 * ket0.dag()
        self.local_P1 = ket1 * ket1.dag()
        self.local_Sp = ket1 * ket0.dag()
        self.local_Sm = ket0 * ket1.dag()
        self.local_X = self.local_Sp + self.local_Sm
        self.local_Y = -1j * (self.local_Sp - self.local_Sm)
        self.local_Z = self.local_P0 - self.local_P1

        self.a_local = qt.destroy(self.Nm)
        self.a = self.phonon(self.a_local)
        self.adag = self.a.dag()
        self.n = self.adag * self.a

    def qubit(self, op, i: int):
        qt = _qutip()
        if not 0 <= i < self.N:
            raise IndexError(f"Qubit index {i} outside [0, {self.N - 1}].")
        ops = [qt.qeye(2) for _ in range(self.N)] + [qt.qeye(self.Nm)]
        ops[i] = op
        return qt.tensor(ops)

    def phonon(self, op):
        qt = _qutip()
        return qt.tensor([qt.qeye(2) for _ in range(self.N)] + [op])

    def X(self, i: int):
        return self.qubit(self.local_X, i)

    def Y(self, i: int):
        return self.qubit(self.local_Y, i)

    def Z(self, i: int):
        return self.qubit(self.local_Z, i)

    def Sp(self, i: int):
        return self.qubit(self.local_Sp, i)

    def Sm(self, i: int):
        return self.qubit(self.local_Sm, i)

    def P0(self, i: int):
        return self.qubit(self.local_P0, i)

    def P1(self, i: int):
        return self.qubit(self.local_P1, i)


class TrappedIonHamiltonian:
    """Effective interaction-picture trapped-ion Hamiltonians."""

    def __init__(self, cfg: TrappedIonConfig, ops: TrappedIonOperators):
        self.cfg = cfg
        self.ops = ops

    def carrier(self, Omega: float = 1.0, delta: float = 0.0, phi: float = 0.0):
        H = 0
        for i in range(self.cfg.N):
            H += 0.5 * Omega * (
                np.exp(1j * phi) * self.ops.Sp(i)
                + np.exp(-1j * phi) * self.ops.Sm(i)
            )
            H += 0.5 * delta * self.ops.Z(i)
        return H

    def red_sideband(self, Omega: float = 1.0, phi: float = 0.0):
        H = 0
        for i in range(self.cfg.N):
            eta_i = self.cfg.eta * self.cfg.mode_weights[i]
            H += 0.5j * eta_i * Omega * (
                np.exp(1j * phi) * self.ops.a * self.ops.Sp(i)
                - np.exp(-1j * phi) * self.ops.adag * self.ops.Sm(i)
            )
        return H

    def blue_sideband(self, Omega: float = 1.0, phi: float = 0.0):
        H = 0
        for i in range(self.cfg.N):
            eta_i = self.cfg.eta * self.cfg.mode_weights[i]
            H += 0.5j * eta_i * Omega * (
                np.exp(1j * phi) * self.ops.adag * self.ops.Sp(i)
                - np.exp(-1j * phi) * self.ops.a * self.ops.Sm(i)
            )
        return H


def computational_state(cfg: TrappedIonConfig, bits: Sequence[int], n_phonon: int = 0):
    qt = _qutip()
    if len(bits) != cfg.N:
        raise ValueError(f"Expected {cfg.N} qubit bits, got {len(bits)}.")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Qubit states must be 0 or 1.")
    if not 0 <= n_phonon < cfg.Nm:
        raise ValueError("n_phonon is outside the phonon cutoff.")
    return qt.tensor([qt.basis(2, bit) for bit in bits] + [qt.basis(cfg.Nm, n_phonon)])


def plus_state(cfg: TrappedIonConfig, n_phonon: int = 0):
    qt = _qutip()
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return qt.tensor([plus for _ in range(cfg.N)] + [qt.basis(cfg.Nm, n_phonon)])


def build_collapse_operators(
    cfg: TrappedIonConfig,
    ops: TrappedIonOperators,
    T1: float = np.inf,
    Tphi: float = np.inf,
    kappa: float = 0.0,
    nbar: float = 0.0,
):
    c_ops = []

    if np.isfinite(T1) and T1 > 0:
        gamma1 = 1.0 / T1
        c_ops.extend(np.sqrt(gamma1) * ops.Sm(i) for i in range(cfg.N))

    if np.isfinite(Tphi) and Tphi > 0:
        gamma_phi = 1.0 / Tphi
        c_ops.extend(np.sqrt(gamma_phi / 2.0) * ops.Z(i) for i in range(cfg.N))

    if kappa > 0:
        c_ops.append(np.sqrt(kappa * (nbar + 1.0)) * ops.a)
        if nbar > 0:
            c_ops.append(np.sqrt(kappa * nbar) * ops.adag)

    return c_ops


def solver_options(store_states: bool = False) -> dict:
    return {
        "store_states": store_states,
        "nsteps": 20000,
        "atol": 1e-10,
        "rtol": 1e-8,
        "progress_bar": False,
    }


def simulate_master(
    H,
    psi0,
    times: np.ndarray,
    observables: Mapping[str, object],
    c_ops: Sequence[object] | None = None,
    store_states: bool = False,
):
    qt = _qutip()
    if c_ops is None:
        c_ops = []
    if not H.isherm:
        raise ValueError("Hamiltonian must be Hermitian.")

    names = list(observables)
    result = qt.mesolve(
        H,
        psi0,
        times,
        c_ops=c_ops,
        e_ops=[observables[name] for name in names],
        options=solver_options(store_states),
    )

    data = {"t": np.asarray(times, dtype=float)}
    for name, values in zip(names, result.expect):
        data[name] = np.real(np.asarray(values))
    if store_states:
        data["states"] = result.states
    return data


def simulate_mc_ensemble(
    H,
    psi0,
    times: np.ndarray,
    c_ops: Sequence[object],
    observable_ops: Sequence[object],
    ntraj: int = 200,
    seed: int = 12345,
):
    qt = _qutip()
    major = int(qt.__version__.split(".")[0])
    if major < 5:
        raise RuntimeError("This project requires QuTiP 5 or newer.")

    options = solver_options(False)
    options["keep_runs_results"] = True

    result = qt.mcsolve(
        H,
        psi0,
        times,
        c_ops=c_ops,
        e_ops=observable_ops,
        ntraj=ntraj,
        seeds=seed,
        options=options,
    )

    if result.runs_expect is None:
        raise RuntimeError("QuTiP did not retain individual trajectories.")

    runs = np.real_if_close(np.asarray(result.runs_expect)).astype(float)
    expected = (len(observable_ops), result.num_trajectories, len(times))
    if runs.shape != expected:
        raise RuntimeError(f"Unexpected trajectory array {runs.shape}; expected {expected}.")
    return result, runs


def extract_jump_metadata(mc_result):
    """Return jump times and collapse-channel indices by trajectory."""

    jump_times = {
        i: np.asarray(times, dtype=float)
        for i, times in enumerate(mc_result.col_times)
    }
    collapse_channels = {
        i: np.asarray(channels, dtype=int)
        for i, channels in enumerate(mc_result.col_which)
    }
    jump_counts = np.asarray([len(times) for times in mc_result.col_times], dtype=int)
    return jump_times, collapse_channels, jump_counts
