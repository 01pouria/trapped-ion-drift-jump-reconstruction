import pytest

qutip = pytest.importorskip("qutip")

import numpy as np

from trapped_ion_pdmp.config import TrappedIonConfig
from trapped_ion_pdmp.quantum import (
    TrappedIonHamiltonian,
    TrappedIonOperators,
    computational_state,
    simulate_master,
)


def test_carrier_matches_analytic_solution():
    cfg = TrappedIonConfig(N=1, Nm=3, eta=0.12)
    ops = TrappedIonOperators(cfg)
    ham = TrappedIonHamiltonian(cfg, ops)

    Omega = 1.0
    t = np.linspace(0, 2 * np.pi, 401)
    result = simulate_master(
        ham.carrier(Omega=Omega),
        computational_state(cfg, [0], 0),
        t,
        {"P_exc": ops.P1(0)},
    )
    exact = np.sin(Omega * t / 2) ** 2
    assert np.max(np.abs(result["P_exc"] - exact)) < 1e-5
