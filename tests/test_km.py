import numpy as np

from trapped_ion_pdmp.km import fit_kernel_neighbor_model, kernel_km_predict


def test_kernel_second_moment_is_psd():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 2))
    dX = 0.01 * rng.normal(size=(500, 2))
    model = fit_kernel_neighbor_model(X, n_neighbors=80)
    result = kernel_km_predict(model, dX, X[:25], tau=0.1)

    eig = np.linalg.eigvalsh(result["D2"])
    assert np.min(eig) > -1e-12


def test_kernel_output_shapes():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 4))
    dX = rng.normal(scale=0.01, size=(200, 4))
    model = fit_kernel_neighbor_model(X, n_neighbors=40)
    result = kernel_km_predict(model, dX, X[:10], tau=0.2)

    assert result["D1"].shape == (10, 4)
    assert result["D2"].shape == (10, 4, 4)
