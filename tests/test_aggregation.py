import numpy as np
import torch

from fedhydra.methods.aggregation import aggregate_fedhydra


def _updates(values: list[float]) -> list[dict[str, torch.Tensor]]:
    return [{"weight": torch.tensor([value])} for value in values]


def test_paper_same_set_aggregation_equals_uniform_mean() -> None:
    responsibilities = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    values = [1.0, 2.0, 7.0, 10.0]
    aggregated, priors = aggregate_fedhydra(
        updates=_updates(values),
        participant_ids=[0, 1, 2, 3],
        responsibilities=responsibilities,
        mixture_weights=responsibilities.mean(axis=0),
        sample_counts=[1, 1, 1, 1],
        epsilon=0.0,
        sample_weighted=False,
        mixture_share_scope="participants",
    )
    assert np.allclose(priors, responsibilities.mean(axis=0))
    assert torch.allclose(
        aggregated["weight"], torch.tensor([np.mean(values)], dtype=torch.float32)
    )


def test_population_priors_make_partial_participation_relation_aware() -> None:
    responsibilities = np.array(
        [[0.99, 0.01], [0.95, 0.05], [0.05, 0.95], [0.01, 0.99]]
    )
    population_weights = responsibilities.mean(axis=0)
    updates = _updates([0.0, 10.0])
    population, _ = aggregate_fedhydra(
        updates,
        [0, 2],
        responsibilities,
        population_weights,
        [1, 1],
        0.0,
        mixture_share_scope="all_clients",
    )
    literal, _ = aggregate_fedhydra(
        updates,
        [0, 2],
        responsibilities,
        population_weights,
        [1, 1],
        0.0,
        mixture_share_scope="participants",
    )
    assert not torch.allclose(population["weight"], literal["weight"])
    assert torch.allclose(literal["weight"], torch.tensor([5.0]))
