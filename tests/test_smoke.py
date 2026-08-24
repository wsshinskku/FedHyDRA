from pathlib import Path

from fedhydra.config import load_config
from fedhydra.federated import FederatedTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_full_smoke_run(tmp_path: Path) -> None:
    config = load_config(
        ROOT / "configs" / "smoke.yaml",
        [
            "federated.rounds=1",
            "federated.checkpoint_every=1",
            "fedhydra.vgae_epochs=1",
        ],
    )
    trainer = FederatedTrainer(config, run_dir=tmp_path / "run")
    summary = trainer.run()
    assert summary["method"] == "fedhydra"
    assert (tmp_path / "run" / "summary.json").is_file()
    assert (tmp_path / "run" / "checkpoints" / "round-0001.pt").is_file()

