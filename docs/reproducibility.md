# Reproducibility notes

## Baseline

- ions: 2
- phonon cutoff: 4
- Lamb--Dicke parameter: 0.15
- Rabi frequency: 1.0
- \(T_1\): 80
- initial state: \(|10,0_{\rm ph}\rangle\)
- trajectories: 200
- time samples: 1001
- kernel neighbors: 400
- random seed: 12345

The observation interval is \(2.2\) coherent-transfer periods, with

\[
t_{\rm transfer} = \frac{2\pi}{\eta\Omega}.
\]

## Train/validation split

Trajectories, not individual increments, are split into training and validation sets. When jump metadata are available, the split is stratified by whether a trajectory contains a jump.

## Continuous sector

The local drift is fitted only on active-origin transitions that do not cross a recorded quantum jump.

The finite-lag second conditional moment is evaluated at lags of 1, 2, 4, and 8 sampling intervals.

## Jump sector

The pooled relaxation-rate estimator is

\[
\hat\gamma =
\frac{N_{\rm jump}}
{\sum_{\rm trajectories}\sum_i\int P_{e,i}(t)\,dt}.
\]

The jump map is estimated from the post-event state of training transitions.

## PDMP integration

The reconstructed deterministic flow is integrated with a midpoint/RK2 update. The state itself is not clipped or projected. The event hazard is evaluated using the physical excited-population range.

## Frozen versus generated outputs

The `results/` directory contains outputs frozen for the manuscript. Full reruns write to `results/generated/`.

This separation prevents a new library version, a changed random-number stream, or a local numerical environment from silently replacing the values cited in the paper.
