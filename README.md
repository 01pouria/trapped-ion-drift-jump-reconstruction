# Data-driven drift--jump reconstruction in a trapped-ion system

Code and frozen outputs for the manuscript:

**Data-driven reconstruction of effective drift--jump dynamics in an open trapped-ion system**

The repository separates the scientific implementation from the notebook used to inspect the published outputs. The original development notebook contained validation gates, duplicated cells, and exploratory checks; those have been removed from the public workflow.

## What the code does

The baseline model contains two trapped ions, one collective motional mode, red-sideband spin--motion coupling, and independent \(T_1\) relaxation. Quantum-jump trajectories are projected onto

\[
X=(P_{e0},P_{e1},C_{sm0},C_{sm1})^T.
\]

Finite-lag conditional moments are estimated locally. No-jump increments are used for the continuous drift, while recorded relaxation events are treated as a separate jump sector. The reconstructed model is a piecewise-deterministic Markov process (PDMP).

The code also includes parameter recovery, held-out reconstruction, a deterministic baseline, a moment-matched Gaussian surrogate, and the appendix-level validation diagnostics.

## Scientific scope

Two implementation details are explicit because they matter for interpreting the claims:

- The quantum-jump event record is used to separate no-jump and jump-containing increments.
- The relaxation hazard has the physics-informed form \(\lambda_i(X)=\gamma P_{e,i}\); the rate \(\gamma\) is estimated from event exposure.

The repository therefore supports a data-driven reconstruction of the effective drift--jump dynamics in the studied benchmark. It is not a blind discovery of arbitrary jump channels, a full Hamiltonian reconstruction, or a universal test against all possible continuous stochastic descriptions.

## Repository layout

```text
.
├── notebooks/
│   └── paper_results.ipynb
├── src/trapped_ion_pdmp/
│   ├── config.py
│   ├── quantum.py
│   ├── observables.py
│   ├── km.py
│   ├── rates.py
│   ├── pdmp.py
│   ├── metrics.py
│   ├── ablation.py
│   ├── statistics.py
│   ├── validation.py
│   ├── pipeline.py
│   └── plotting.py
├── scripts/
│   ├── reproduce_full.py
│   ├── run_parameter_sweeps.py
│   ├── run_robustness.py
│   ├── run_validation.py
│   └── check_frozen_results.py
├── tests/
├── results/
│   ├── figures/
│   ├── tables/
│   ├── supplementary/
│   ├── frozen_metrics.json
│   └── manifest.json
└── data/
```

## Quick start

Create a clean environment with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

Open the compact results notebook:

```bash
jupyter lab notebooks/paper_results.ipynb
```

The notebook reads the frozen tables and figures already stored in `results/`. It does not define the simulation engine inline.

## Full reproduction

A full baseline rerun requires QuTiP and can take noticeably longer than the quick notebook:

```bash
python scripts/reproduce_full.py
```

Newly generated outputs are written to:

```text
results/generated/
```

The frozen manuscript outputs are never overwritten automatically.

Parameter sweeps can be rerun separately:

```bash
python scripts/run_parameter_sweeps.py
```

Robustness and appendix diagnostics:

```bash
python scripts/run_robustness.py
python scripts/run_validation.py
```

## Tests

```bash
pytest
```

Tests that require QuTiP are skipped automatically when QuTiP is not installed.

## Frozen values and numerical regression

`results/frozen_metrics.json` records the values used in the current manuscript. `scripts/check_frozen_results.py` verifies that the main CSV tables still match those values.

The baseline point estimate of \(g\) reported in the manuscript is `0.053021`. The mean of the trajectory-bootstrap distribution is `0.053031`; these are different statistical summaries and are kept distinct.

## Data

The raw trajectory ensemble is not duplicated in the repository. It can be regenerated from the fixed seeds and model parameters. Processed manuscript tables and figure outputs are included under `results/`.

See `data/README.md` and `docs/reproducibility.md` for details.

## Citation

Citation metadata is provided in `CITATION.cff`. Add the final journal DOI there after publication.

## License

MIT. See `LICENSE`.
