import numpy as np

from trapped_ion_pdmp.metrics import diagnose_state_domain, survival_curve


def test_survival_curve():
    jumps = np.array([1.0, 2.0, np.nan, 3.0])
    times = np.array([0.0, 1.5, 2.5, 4.0])
    curve = survival_curve(jumps, times)
    np.testing.assert_allclose(curve, [1.0, 0.75, 0.5, 0.25])


def test_domain_diagnostic():
    states = np.zeros((2, 3, 4))
    states[..., 0] = 0.4
    states[..., 1] = 0.3
    clean = diagnose_state_domain(states)
    assert clean["fraction"] == 0.0

    states[0, 0, 0] = 1.2
    bad = diagnose_state_domain(states)
    assert bad["fraction"] > 0.0
    assert bad["max_violation"] >= 0.2 - 1e-12
