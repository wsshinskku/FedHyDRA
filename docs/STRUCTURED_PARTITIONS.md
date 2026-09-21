# Structured client partitions

The structured partition combines persistent client groups, adjacent label overlap, boundary clients, and label-preserving feature shifts. The CIFAR-100 profile uses five groups and adjacent overlap ratios of 20%, 20%, 30%, and 10%.

## Generation rules

`structured_profiles()` applies these seeded rules:

1. Split class IDs into `structured_groups` disjoint anchor blocks.
2. Allocate each group's class mass to its own anchor and adjacent anchors.
3. Assign clients evenly to persistent groups and shuffle assignments.
4. Select `boundary_fraction` clients without replacement and blend each with an adjacent group using a weight in `[0.35, 0.65]`.
5. Draw final class profiles with the configured Dirichlet concentration.
6. Allocate a fixed number of examples to each client without replacement. Redistribute an exhausted class's remaining demand over available classes.
7. Apply a convex mixture of deterministic group color transforms to create conditional feature shift while preserving labels.

Each `partition.json` records the exact sample indices, class profiles, domain memberships, group assignments, and boundary clients. Use the same seed and partition settings across compared methods.

## Custom partitions

Construct a `fedhydra.data.partition.Partition` with:

- `client_indices`: one integer NumPy array per client;
- `domain_memberships`: an `N x G` row-stochastic array;
- `class_profiles`: an `N x C` row-stochastic array;
- `metadata`: JSON-serializable provenance.

Call `partition.validate(dataset_size, minimum)` before training. To integrate a saved external split, load it into this object at the `build_partition()` call in `FederatedTrainer`.
