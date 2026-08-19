#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trapped_ion_pdmp.pipeline import (
    fit_pdmp_components,
    reconstruct_and_evaluate,
    simulate_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Generating the baseline quantum-jump ensemble...")
    baseline = simulate_baseline()

    print("Fitting the drift, relaxation rate, and jump map...")
    fitted = fit_pdmp_components(baseline)

    print("Running the independent PDMP reconstruction...")
    reconstruction, evaluation = reconstruct_and_evaluate(
        baseline,
        fitted,
        ntraj=300,
        substeps=1,
    )

    baseline.dataframe.to_csv(OUT / "baseline_trajectory_dataframe.csv.gz", index=False)
    pd.DataFrame([evaluation["summary"]]).to_csv(
        OUT / "reconstruction_summary.csv",
        index=False,
    )
    evaluation["moments"].to_csv(OUT / "reconstruction_moments.csv", index=False)
    evaluation["wasserstein"].to_csv(
        OUT / "reconstruction_wasserstein.csv",
        index=False,
    )

    np.savez_compressed(
        OUT / "reconstruction_arrays.npz",
        times=baseline.times,
        states=reconstruction["states"],
        jump_times=reconstruction["jump_times"],
        quantum_validation=fitted["quantum_validation"],
        quantum_first_jump=fitted["quantum_first_jump"],
    )

    metadata = {
        "train_trajectories": len(fitted["train_ids"]),
        "validation_trajectories": len(fitted["validation_ids"]),
        "continuous_train_transitions": len(fitted["dX_drift_train"]),
        "gamma_hat": fitted["gamma_hat"],
        "jump_target": fitted["jump_target"].tolist(),
    }
    (OUT / "fit_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(pd.DataFrame([evaluation["summary"]]).to_string(index=False))
    print(f"\nGenerated files: {OUT}")


if __name__ == "__main__":
    main()
