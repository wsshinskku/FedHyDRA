import numpy as np

from fedhydra.config import DataConfig
from fedhydra.data.partition import build_partition


def test_structured_partition_is_fixed_budget_disjoint_and_deterministic() -> None:
    labels = np.tile(np.arange(6), 40)
    config = DataConfig(
        name="synthetic",
        num_clients=6,
        clients_per_round=3,
        partition="structured",
        samples_per_client=20,
        min_samples_per_client=5,
        structured_groups=3,
        overlap_ratios=[0.2, 0.3],
        boundary_fraction=0.5,
    )
    first = build_partition(labels, config, seed=9)
    second = build_partition(labels, config, seed=9)
    assert all(len(indices) == 20 for indices in first.client_indices)
    flattened = np.concatenate(first.client_indices)
    assert len(flattened) == len(np.unique(flattened))
    assert np.allclose(first.domain_memberships.sum(axis=1), 1.0)
    for left, right in zip(first.client_indices, second.client_indices, strict=True):
        assert np.array_equal(left, right)


def test_dirichlet_profiles_are_not_uniform() -> None:
    labels = np.tile(np.arange(4), 50)
    config = DataConfig(
        num_clients=5,
        clients_per_round=2,
        partition="dirichlet",
        alpha=0.3,
        samples_per_client=20,
        min_samples_per_client=5,
    )
    partition = build_partition(labels, config, seed=1)
    assert partition.class_profiles.shape == (5, 4)
    assert not np.allclose(partition.class_profiles, 0.25)

