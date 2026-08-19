#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from trapped_ion_pdmp.config import PARAMETRIC_CASES
from trapped_ion_pdmp.pipeline import simulate_parametric_case
from trapped_ion_pdmp.rates import recover_coupling, recover_gamma

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, (name, eta, Omega, T1) in enumerate(PARAMETRIC_CASES):
        print(f"Running {name}...")
        case = simulate_parametric_case(
            name,
            eta,
            Omega,
            T1,
            ntraj=120,
            n_time=801,
            seed=12345 + 1000 + i,
        )
        coupling = recover_coupling(
            case["dataframe"],
            case["jump_times"],
            eta,
            Omega,
            case["cfg"].mode_weights,
            seed=12345 + 2000 + i,
        )
        gamma = recover_gamma(
            case["dataframe"],
            case["collapse_channels"],
            T1,
            coupling["train_ids"],
            coupling["validation_ids"],
        )

        rows.append(
            {
                "case": name,
                "eta": eta,
                "Omega": Omega,
                "T1": T1,
                "g_true": coupling["g_true"],
                "g0_hat": coupling["g0_hat"],
                "g1_hat": coupling["g1_hat"],
                "g_hat": coupling["g_hat"],
                "g_relative_error": coupling["g_relative_error"],
                "drift_NRMSE": coupling["drift_NRMSE"],
                "gamma_true": gamma["gamma_true"],
                "gamma_hat": gamma["gamma_hat"],
                "gamma_relative_error": gamma["gamma_relative_error"],
                "train_jumps": gamma["train_jumps"],
                "validation_jumps": gamma["validation_jumps"],
                "predicted_validation_jumps": gamma["predicted_validation_jumps"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "parametric_cases.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
