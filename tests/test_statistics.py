import numpy as np

from trapped_ion_pdmp.statistics import poisson_rate_ci, wilson_interval


def test_poisson_interval_contains_mle():
    low, high = poisson_rate_ci(12, 100.0)
    assert low < 0.12 < high


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(60, 100)
    assert 0.0 < low < 0.6 < high < 1.0
