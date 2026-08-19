# Scientific scope

This repository is intentionally narrower than a general open-quantum-system inference package.

## Inferred from data

- The reduced-state drift is estimated from no-jump trajectory increments.
- The relaxation rate is estimated from event counts and state exposure.
- The post-jump map is estimated from jump-containing training transitions.
- Held-out trajectory statistics are generated from the inferred effective model.

## Supplied by the benchmark design

- The system is a two-ion red-sideband model.
- The available jump record comes from the quantum-jump unraveling.
- The relaxation hazard uses the form \(\lambda_i(X)=\gamma P_{e,i}\).
- The selected observables are fixed before the held-out reconstruction.

## Claims not supported by this code

The implementation should not be used to claim:

- full Hamiltonian reconstruction;
- blind discovery of unknown jump channels;
- exact Markovianity of the reduced state;
- universal absence of quantum diffusion;
- superiority of PDMPs over every continuous stochastic model;
- exact global recovery of the observable manifold.

The finite-lag second conditional moment is interpreted only over the sampled lags and for the present unraveling and parameter regime.
