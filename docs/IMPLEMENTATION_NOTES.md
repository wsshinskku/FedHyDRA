# Implementation notes

This guide maps the FedHyDRA equations to the official implementation and describes the numerical conventions used by its configuration profiles.

## Equation-to-module map

Paths below are relative to `src/fedhydra/`.

| Paper | Implementation |
|---|---|
| Eqs. (1)–(2), local SGD and delta | `federated/client.py` |
| Eqs. (3)–(4), histogram and RFF mean | `methods/summaries.py` |
| Eqs. (6)–(12), JSD/MMD/calibration/omega | `methods/relations.py` |
| Eqs. (13)–(15), top-k graph and node features | `methods/relations.py`, `federated/server.py` |
| Eqs. (16)–(18), VGAE | `methods/vgae.py` |
| Eqs. (19)–(21), full GMM and WSS | `methods/gmm.py` |
| Eqs. (22)–(23), aggregation | `methods/aggregation.py` |
| Algorithm 1 cadence order | `federated/server.py` |

## Round semantics

All selected clients compute summaries from the round-start global feature extractor and train local models from that shared starting state. The server refreshes state in this order:

```text
hybrid weight -> graph/VGAE -> GMM -> aggregation
```

Cached state is reused between refresh boundaries. Before the first round, a summary-only pass initializes the graph over every client. This bootstrap pass is outside the timed communication rounds.

## Random Fourier features

A shared Gaussian projection and uniform phase approximate the RBF kernel:

```text
k(x, y) = exp(-gamma * ||x-y||^2)
```

The projection seed is `experiment.seed + 101`. RFF dimension, gamma, and optional L2 feature normalization are configurable. The CIFAR-100 profile uses 256 features, `gamma=1.0`, and `normalize_features: false`.

## Variational graph autoencoder

The dense VGAE uses:

- weighted symmetric adjacency and self-loops for encoder message passing;
- symmetric degree normalization;
- a shared ReLU GCN followed by mean and log-standard-deviation GCN heads;
- an inner-product decoder with binary off-diagonal graph targets;
- optional positive-class weighting, disabled in the CIFAR-100 profile;
- posterior samples during training and posterior means for GMM input;
- Adam, gradient clipping, and configurable KL weight and epochs.

## Gaussian mixture model

The full-covariance `sklearn.mixture.GaussianMixture` uses five k-means initializations, covariance regularization, and a seeded fit on the current embeddings at each refresh. Responsibilities are row-normalized. Mixture shares are the mean responsibilities, as in Eq. (20).

## Partial-participation aggregation

The default `all_clients` mode uses population mixture shares from cached client memberships and cluster means from the currently selected client updates.

The `participants` mode computes both quantities over the same active set. For the same set of `N` clients, with no sample weighting and zero epsilon,

```text
pi_k = (1/N) sum_i Gamma_ik
cluster_k = sum_i Gamma_ik Delta_i / sum_i Gamma_ik
sum_k pi_k cluster_k = (1/N) sum_i Delta_i
```

This identity gives a useful aggregation invariant for tests and comparisons. Record `mixture_share_scope`, sample weighting, and epsilon with results.

## Model and local training

The image model is MobileNetV2 with width multiplier 0.5, a stride pattern for 32–96 pixel inputs, and a 512-dimensional penultimate layer. Aggregation includes floating-point state, including BatchNorm running statistics; integer buffers retain the server values.

Cosine learning-rate decay is indexed by communication round. Clients in a round share the same learning rate and each local optimization run initializes a fresh optimizer.

## Metrics

Balanced top-1 accuracy averages per-class recall over classes present in the test split. Convergence is the first evaluated round reaching 90% of final evaluated accuracy.

CoV and Min/Mean use all evaluation points with population standard deviation. The metric helper also supports an explicit `burn_in` argument; default run summaries use zero. Round timing covers training, summaries, refreshes, and aggregation, with evaluation outside the timed window.
