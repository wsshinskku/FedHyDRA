# Implementation notes

This document maps the manuscript equations to code and identifies every material completion made by the repository.

## Equation-to-module map

| Manuscript | Implementation |
|---|---|
| Eqs. (1)-(2), local SGD and delta | `federated/client.py` |
| Eqs. (3)-(4), histogram and RFF mean | `methods/summaries.py` |
| Eqs. (6)-(12), JSD/MMD/calibration/omega | `methods/relations.py` |
| Eqs. (13)-(15), top-k graph and node features | `methods/relations.py`, `federated/server.py` |
| Eqs. (16)-(18), VGAE | `methods/vgae.py` |
| Eqs. (19)-(21), full GMM and WSS | `methods/gmm.py` |
| Eqs. (22)-(23), aggregation | `methods/aggregation.py` |
| Algorithm 1 cadence order | `federated/server.py` |

## Round semantics

At the start of every round, the global model is unchanged while all selected clients compute their summaries and local models. RFF features therefore come from the round-start global feature extractor, never a locally updated extractor. The server then refreshes state in the required order:

```text
hybrid weight -> graph/VGAE -> GMM -> aggregation
```

Cached state is reused between cadence boundaries.

The first graph requires summaries for all clients although round zero has only 20 participants. The implementation performs one summary-only bootstrap pass over every client. This cost is outside timed rounds and is recorded as an explicit assumption.

## RFF completion

The manuscript specifies only a shared `D`-dimensional RFF map. The implementation approximates

```text
k(x, y) = exp(-gamma * ||x-y||^2)
```

with a fixed Gaussian projection and uniform phase. Penultimate features are L2-normalized by default, `gamma=1.0`, and the projection seed is `experiment.seed + 101`. All are configurable.

## VGAE completion

The graph has only 200 nodes, so a dense implementation is clearer and avoids an additional graph framework. It uses:

- weighted symmetric adjacency plus self-loops for encoder message passing;
- symmetric degree normalization;
- one shared ReLU GCN and separate mean/log-standard-deviation GCN heads;
- an inner-product decoder trained against binary off-diagonal graph support;
- optional positive-class weighting (disabled in the manuscript-default config);
- posterior samples for the ELBO and posterior means for the GMM;
- Adam, gradient clipping, and configurable KL weight/epochs.

These are standard engineering choices, not parameters reported by the paper.

## GMM completion

`sklearn.mixture.GaussianMixture` is configured with full covariance, five k-means initializations, covariance regularization, and a deterministic seed. Each refresh is fit deterministically from the current embeddings rather than depending on an estimator state that cannot be represented by the paper equations. Responsibilities are normalized again defensively. Mixture weights used by the aggregation are the responsibility mean, exactly matching Eq. (20).

## Aggregation identity

For the same set of `N` clients, no sample weighting, and zero epsilon:

```text
pi_k = (1/N) sum_i Gamma_ik
cluster_k = sum_i Gamma_ik Delta_i / sum_i Gamma_ik
```

implies

```text
sum_k pi_k cluster_k = (1/N) sum_i Delta_i.
```

Therefore graph/GMM state cannot alter the update under a literal same-set reading. The default `all_clients` mode follows the most operationally consistent partial-participation interpretation: population mixture weights from all 200 cached clients and cluster means from the 20 current updates. The alternative `participants` mode makes the identity observable.

## Model and training completion

“Lightweight MobileNet” is implemented as a self-contained MobileNetV2 with width multiplier 0.5, a stride pattern suitable for 32-96 pixel inputs, and a 512-dimensional penultimate layer. The code aggregates all floating state, including BatchNorm running statistics, and retains integer buffers from the server model.

Cosine decay is indexed by communication round. Every client in a round receives the same learning rate and initializes its optimizer afresh, which is a conventional synchronous FL interpretation.

## Metrics

Balanced top-1 accuracy is the arithmetic mean of per-class recall over classes present in the test split. Convergence is the first evaluated round reaching 90% of the final evaluated accuracy.

CoV and Min-Mean are computed over all evaluation points with population standard deviation. The paper does not give a burn-in or trailing window. The metric helper supports an explicit `burn_in` argument for sensitivity analysis, but the default run summary uses zero.
