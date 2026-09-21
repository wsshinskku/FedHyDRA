"""Fixed-budget IID, Dirichlet, and structured overlapping partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fedhydra.config import DataConfig


@dataclass(slots=True)
class Partition:
    client_indices: list[np.ndarray]
    domain_memberships: np.ndarray
    class_profiles: np.ndarray
    metadata: dict[str, Any]

    def validate(self, dataset_size: int, minimum: int = 1) -> None:
        if len(self.client_indices) != self.domain_memberships.shape[0]:
            raise ValueError("client count and domain membership count differ")
        flattened = np.concatenate(self.client_indices) if self.client_indices else np.array([])
        if flattened.size != np.unique(flattened).size:
            raise ValueError("a training sample was assigned to more than one client")
        if flattened.size and (flattened.min() < 0 or flattened.max() >= dataset_size):
            raise ValueError("partition contains an out-of-range sample index")
        if any(len(indices) < minimum for indices in self.client_indices):
            raise ValueError("at least one client is below the minimum sample count")
        if not np.allclose(self.domain_memberships.sum(axis=1), 1.0):
            raise ValueError("domain memberships must sum to one per client")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "client_indices": [indices.tolist() for indices in self.client_indices],
            "domain_memberships": self.domain_memberships.tolist(),
            "class_profiles": self.class_profiles.tolist(),
            "metadata": self.metadata,
        }


def _budget(config: DataConfig, dataset_size: int) -> int:
    value = config.samples_per_client or dataset_size // config.num_clients
    if value < config.min_samples_per_client:
        raise ValueError(
            f"samples_per_client={value} is below min_samples_per_client="
            f"{config.min_samples_per_client}"
        )
    if value * config.num_clients > dataset_size:
        raise ValueError(
            "fixed client budgets require more samples than the training dataset contains"
        )
    return int(value)


def _iid_partition(dataset_size: int, clients: int, budget: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_size)[: clients * budget]
    return [
        np.sort(indices[client * budget : (client + 1) * budget]) for client in range(clients)
    ]


def _draw_profiles(
    base_profiles: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if concentration <= 0:
        return base_profiles.copy()
    parameters = np.maximum(base_profiles * concentration, 1.0e-3)
    return np.stack([rng.dirichlet(row) for row in parameters])


def _fixed_budget_allocation(
    labels: np.ndarray,
    profiles: np.ndarray,
    budget: int,
    seed: int,
) -> list[np.ndarray]:
    """Allocate without replacement while honoring per-client class profiles."""

    rng = np.random.default_rng(seed)
    classes = profiles.shape[1]
    pools: list[list[int]] = []
    for label in range(classes):
        values = np.flatnonzero(labels == label)
        pools.append(rng.permutation(values).tolist())

    result: list[np.ndarray | None] = [None] * len(profiles)
    for client in rng.permutation(len(profiles)).tolist():
        profile = np.asarray(profiles[client], dtype=np.float64)
        profile = np.maximum(profile, 0)
        profile /= profile.sum()
        desired = rng.multinomial(budget, profile)
        chosen: list[int] = []
        for label, count in enumerate(desired.tolist()):
            take = min(count, len(pools[label]))
            if take:
                chosen.extend(pools[label][-take:])
                del pools[label][-take:]

        while len(chosen) < budget:
            available = np.asarray([len(pool) > 0 for pool in pools], dtype=bool)
            if not available.any():
                raise RuntimeError("ran out of samples while constructing fixed client budgets")
            probabilities = profile * available
            if probabilities.sum() <= 0:
                probabilities = available.astype(np.float64)
            probabilities /= probabilities.sum()
            label = int(rng.choice(classes, p=probabilities))
            chosen.append(pools[label].pop())
        result[client] = np.sort(np.asarray(chosen, dtype=np.int64))

    return [indices for indices in result if indices is not None]


def _edge_ratios(groups: int, values: list[float]) -> np.ndarray:
    ratios = np.zeros(max(groups - 1, 0), dtype=np.float64)
    if not values:
        return ratios
    for edge in range(groups - 1):
        ratios[edge] = values[min(edge, len(values) - 1)]
    return ratios


def structured_profiles(
    classes: int,
    clients: int,
    groups: int,
    overlap_ratios: list[float],
    boundary_fraction: float,
    concentration: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate persistent group profiles with adjacent overlap and boundary clients.

    The seeded template assigns disjoint anchor class blocks to groups, adds
    adjacent overlap mass, and interpolates a seeded fraction of boundary clients.
    """

    rng = np.random.default_rng(seed)
    anchors = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(np.arange(classes), groups)]
    edge_ratios = _edge_ratios(groups, overlap_ratios)
    group_profiles = np.zeros((groups, classes), dtype=np.float64)
    for group in range(groups):
        left = edge_ratios[group - 1] if group > 0 else 0.0
        right = edge_ratios[group] if group < groups - 1 else 0.0
        own = max(1.0 - left - right, 0.05)
        group_profiles[group, anchors[group]] += own / max(len(anchors[group]), 1)
        if group > 0:
            group_profiles[group, anchors[group - 1]] += left / max(len(anchors[group - 1]), 1)
        if group < groups - 1:
            group_profiles[group, anchors[group + 1]] += right / max(len(anchors[group + 1]), 1)
        group_profiles[group] /= group_profiles[group].sum()

    assignments = np.arange(clients) % groups
    rng.shuffle(assignments)
    memberships = np.eye(groups, dtype=np.float64)[assignments]
    boundary_count = int(round(clients * boundary_fraction))
    boundary_clients = rng.choice(clients, size=boundary_count, replace=False)
    for client in boundary_clients.tolist():
        group = int(assignments[client])
        candidates = [candidate for candidate in (group - 1, group + 1) if 0 <= candidate < groups]
        if candidates:
            neighbor = int(rng.choice(candidates))
            blend = float(rng.uniform(0.35, 0.65))
            memberships[client] *= blend
            memberships[client, neighbor] += 1.0 - blend

    bases = memberships @ group_profiles
    profiles = _draw_profiles(bases, concentration, rng)
    metadata = {
        "template": "adjacent_anchor_blocks_v1",
        "edge_overlap_ratios": edge_ratios.tolist(),
        "boundary_clients": sorted(int(value) for value in boundary_clients.tolist()),
        "anchor_classes": [values.tolist() for values in anchors],
        "group_assignments": assignments.tolist(),
        "note": "Seeded adjacent-anchor partition with soft boundary-client memberships.",
    }
    return profiles, memberships, metadata


def build_partition(labels: np.ndarray, config: DataConfig, seed: int) -> Partition:
    labels = np.asarray(labels, dtype=np.int64)
    classes = int(labels.max()) + 1
    if config.partition == "structured" and config.structured_groups > classes:
        raise ValueError("structured_groups cannot exceed the number of dataset classes")
    budget = _budget(config, len(labels))
    rng = np.random.default_rng(seed)

    if config.partition == "iid":
        profiles = np.full((config.num_clients, classes), 1.0 / classes)
        memberships = np.ones((config.num_clients, 1), dtype=np.float64)
        indices = _iid_partition(len(labels), config.num_clients, budget, seed)
        metadata: dict[str, Any] = {"partition": "iid"}
    elif config.partition == "dirichlet":
        profiles = rng.dirichlet(
            np.full(classes, config.alpha, dtype=np.float64), size=config.num_clients
        )
        memberships = np.ones((config.num_clients, 1), dtype=np.float64)
        indices = _fixed_budget_allocation(labels, profiles, budget, seed + 1)
        metadata = {"partition": "dirichlet", "alpha": config.alpha}
    else:
        profiles, memberships, metadata = structured_profiles(
            classes=classes,
            clients=config.num_clients,
            groups=config.structured_groups,
            overlap_ratios=config.overlap_ratios,
            boundary_fraction=config.boundary_fraction,
            concentration=config.profile_concentration,
            seed=seed,
        )
        indices = _fixed_budget_allocation(labels, profiles, budget, seed + 1)
        metadata["partition"] = "structured"

    metadata.update(
        {
            "seed": seed,
            "num_clients": config.num_clients,
            "samples_per_client": budget,
            "assigned_samples": config.num_clients * budget,
            "dataset_samples": len(labels),
        }
    )
    realized = np.stack(
        [
            np.bincount(labels[client_indices], minlength=classes) / len(client_indices)
            for client_indices in indices
        ]
    )
    metadata["realized_class_histograms"] = realized.tolist()
    metadata["target_realized_l1_mean"] = float(np.abs(profiles - realized).sum(axis=1).mean())
    partition = Partition(indices, memberships, profiles, metadata)
    partition.validate(len(labels), config.min_samples_per_client)
    return partition
