from pathlib import Path

import pytest
import torch

from fedhydra.config import load_config
from fedhydra.federated import FederatedTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_resume_matches_continuous_training(tmp_path: Path) -> None:
    config = load_config(
        ROOT / "configs" / "smoke.yaml",
        [
            "federated.rounds=2",
            "federated.checkpoint_every=1",
            "fedhydra.vgae_epochs=1",
        ],
    )
    continuous = FederatedTrainer(config, run_dir=tmp_path / "continuous")
    continuous.run()
    round_one = tmp_path / "continuous" / "checkpoints" / "round-0001.pt"

    resumed = FederatedTrainer(
        config,
        run_dir=tmp_path / "resumed",
        resume=round_one,
    )
    resumed.run()
    for name, value in continuous.model.state_dict().items():
        assert torch.equal(value.cpu(), resumed.model.state_dict()[name].cpu())
    assert continuous.server is not None and resumed.server is not None
    assert continuous.server.omega == resumed.server.omega
    assert torch.equal(
        torch.from_numpy(continuous.server.responsibilities),
        torch.from_numpy(resumed.server.responsibilities),
    )


def test_nonempty_run_directory_requires_resume(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        FederatedTrainer(config, run_dir=occupied)

