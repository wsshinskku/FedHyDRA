"""Memory-safe named-tensor aggregation for FedAvg and FedHyDRA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

TensorMap = Mapping[str, torch.Tensor]


def model_delta(
    local_state: TensorMap,
    global_state: TensorMap,
) -> dict[str, torch.Tensor]:
    delta: dict[str, torch.Tensor] = {}
    for name, global_value in global_state.items():
        if global_value.is_floating_point() or global_value.is_complex():
            delta[name] = local_state[name].detach().cpu() - global_value.detach().cpu()
    return delta


def aggregate_fedavg(
    updates: Sequence[TensorMap],
    sample_counts: Sequence[int],
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("cannot aggregate an empty update list")
    weights = np.asarray(sample_counts, dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("sample counts must be positive")
    weights /= weights.sum()
    result: dict[str, torch.Tensor] = {}
    for name in updates[0]:
        accumulator = torch.zeros_like(updates[0][name])
        for weight, update in zip(weights.tolist(), updates, strict=True):
            accumulator.add_(update[name], alpha=weight)
        result[name] = accumulator
    return result


def aggregate_fedhydra(
    updates: Sequence[TensorMap],
    participant_ids: Sequence[int],
    responsibilities: np.ndarray,
    mixture_weights: np.ndarray,
    sample_counts: Sequence[int],
    epsilon: float,
    sample_weighted: bool = False,
    mixture_share_scope: str = "all_clients",
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Implement Eqs. (20)-(23) under an explicit partial-participation scope.

    ``all_clients`` uses cached population mixture shares from Eq. (20) and
    participant-only cluster updates. ``participants`` recomputes shares from
    participants; with epsilon=0 and no sample weighting this algebraically
    reduces to the uniform mean, an identity tested by this repository.
    """

    if not updates or len(updates) != len(participant_ids):
        raise ValueError("updates and participant IDs must be non-empty and aligned")
    participant_ids = np.asarray(participant_ids, dtype=np.int64)
    memberships = np.asarray(responsibilities, dtype=np.float64)[participant_ids]
    if not np.allclose(memberships.sum(axis=1), 1.0, atol=1.0e-6):
        raise ValueError("each responsibility row must sum to one")
    if mixture_share_scope == "participants":
        priors = memberships.mean(axis=0)
    elif mixture_share_scope == "all_clients":
        priors = np.asarray(mixture_weights, dtype=np.float64)
    else:
        raise ValueError("mixture_share_scope must be all_clients or participants")
    priors = priors / priors.sum()

    contribution = memberships.copy()
    if sample_weighted:
        counts = np.asarray(sample_counts, dtype=np.float64)
        contribution *= counts[:, None]
    denominators = contribution.sum(axis=0) + epsilon
    clusters = contribution.shape[1]

    result: dict[str, torch.Tensor] = {}
    for name in updates[0]:
        global_accumulator = torch.zeros_like(updates[0][name])
        for cluster in range(clusters):
            cluster_update = torch.zeros_like(updates[0][name])
            if denominators[cluster] <= 0:
                continue
            for client, update in enumerate(updates):
                cluster_update.add_(
                    update[name], alpha=float(contribution[client, cluster])
                )
            cluster_update.div_(float(denominators[cluster]))
            global_accumulator.add_(cluster_update, alpha=float(priors[cluster]))
        result[name] = global_accumulator
    return result, priors


def apply_delta(
    global_state: TensorMap,
    delta: TensorMap,
    server_learning_rate: float,
) -> dict[str, torch.Tensor]:
    updated: dict[str, torch.Tensor] = {}
    for name, value in global_state.items():
        value = value.detach().cpu()
        if name in delta:
            updated[name] = value + delta[name] * server_learning_rate
        else:
            updated[name] = value.clone()
    return updated

