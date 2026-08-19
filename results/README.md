# Results

The files in `tables/`, `figures/`, and `supplementary/` are the frozen manuscript outputs extracted from the validated development run.

Do not overwrite these files during routine reruns. New calculations belong in `results/generated/`.

`frozen_metrics.json` distinguishes:

- the final 300-trajectory held-out PDMP reconstruction;
- the matched-sample PDMP used in the ablation;
- the baseline manuscript point estimates.

This distinction avoids mixing values obtained from ensembles with different sample sizes.
