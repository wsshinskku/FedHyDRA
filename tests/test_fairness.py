from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from fedhydra.config import FederatedConfig, load_config
from fedhydra.federated.client import train_client
from fedhydra.federated.trainer import FederatedTrainer

ROOT = Path(__file__).resolve().parents[1]


class DropoutClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(12, 2)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.flatten(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.extract_features(inputs)))


def test_client_seed_is_independent_of_prior_global_rng_use() -> None:
    torch.manual_seed(5)
    model = DropoutClassifier()
    dataset = TensorDataset(torch.randn(8, 3, 2, 2), torch.arange(8) % 2)
    config = FederatedConfig(
        rounds=1,
        local_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        momentum=0.0,
        weight_decay=0.0,
        amp=False,
    )
    first = train_client(0, model, dataset, config, torch.device("cpu"), 0, 1, 77, 0)
    _ = torch.randn(10_000)
    second = train_client(0, model, dataset, config, torch.device("cpu"), 0, 1, 77, 0)
    for name in first.update:
        assert torch.equal(first.update[name], second.update[name])


def test_fedavg_never_computes_structural_summaries(tmp_path: Path) -> None:
    config = load_config(
        ROOT / "configs" / "smoke.yaml",
        ["federated.rounds=1", "federated.method=fedavg"],
    )
    trainer = FederatedTrainer(config, run_dir=tmp_path / "fedavg")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("FedAvg must not compute FedHyDRA summaries")

    trainer._update_summary = fail_if_called  # type: ignore[method-assign]
    summary = trainer.run()
    assert summary["method"] == "fedavg"

