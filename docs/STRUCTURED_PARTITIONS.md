# Structured partition benchmark

## Why a completion is necessary

The manuscript describes five CIFAR-100 client groups through background, object, and animal factors, with adjacent overlap percentages of 20%, 20%, 30%, and 10%. It does not define mappings from these factors to training indices or transformations. CIFAR-100 does not provide background annotations for “forest,” “city,” or “roads.” The Tiny-ImageNet and STL-10 templates are not enumerated.

The exact reported split is therefore not reconstructible from the manuscript alone.

## Included deterministic template

`structured_profiles()` uses the following executable rules:

1. Split class IDs into `structured_groups` disjoint anchor blocks.
2. Give each group its own anchor mass and the configured mass on adjacent anchors.
3. Assign clients evenly to persistent groups, then shuffle with the partition seed.
4. Choose `boundary_fraction` clients without replacement and interpolate each between its group and an adjacent group by a seeded weight in `[0.35, 0.65]`.
5. Draw the final class profile around that base profile with a configurable Dirichlet concentration.
6. Allocate a fixed number of examples to every client without replacement. If a desired class is exhausted, redistribute only the deficit over available classes.
7. Apply a deterministic convex mixture of group color transforms to generate conditional feature shift without changing labels.

The generated `partition.json` contains every sample index, client class profile, domain membership, group assignment, and boundary client. Reusing this file across methods is the fairness control.

## Plugging in an original partition

Construct a `fedhydra.data.partition.Partition` with:

- `client_indices`: one integer NumPy array per client;
- `domain_memberships`: an `N x G` row-stochastic array;
- `class_profiles`: an `N x C` row-stochastic array;
- `metadata`: JSON-serializable provenance.

Call `partition.validate(dataset_size, minimum)` before training. A private manifest loader can then replace the call to `build_partition()` in `FederatedTrainer` without changing any algorithm code.

