#!/usr/bin/env python
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def assert_close(label, value, expected, atol=5e-7):
    if not np.isclose(value, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {value} != {expected}")


def main():
    frozen = json.loads((RESULTS / "frozen_metrics.json").read_text())
    table2 = pd.read_csv(RESULTS / "tables" / "Table_2_Physical_Parameter_Recovery.csv")
    table4 = pd.read_csv(RESULTS / "tables" / "Table_4_Model_Ablation.csv")

    g = table2.loc[table2["Parameter"] == "g"].iloc[0]
    gamma = table2.loc[table2["Parameter"] == "gamma"].iloc[0]
    assert_close("g estimate", g["Estimate"], frozen["manuscript_baseline"]["g_estimate"])
    assert_close(
        "gamma estimate",
        gamma["Estimate"],
        frozen["manuscript_baseline"]["gamma_estimate"],
    )

    pdmp = table4.loc[table4["model"] == "PDMP"].iloc[0]
    ablation = frozen["ablation_pdmp_matched_sample"]
    for key in ("mean_NRMSE", "variance_NRMSE", "wasserstein", "survival_RMSE"):
        assert_close(f"ablation {key}", pdmp[key], ablation[key])

    print("Frozen manuscript tables are internally consistent.")


if __name__ == "__main__":
    main()
