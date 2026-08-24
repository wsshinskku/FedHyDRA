from pathlib import Path

from fedhydra.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_config_and_override() -> None:
    config = load_config(
        ROOT / "configs" / "cifar100-random.yaml",
        ["experiment.seed=3", "fedhydra.graph_neighbors=10"],
    )
    assert config.data.partition == "dirichlet"
    assert config.data.alpha == 0.3
    assert config.experiment.seed == 3
    assert config.fedhydra.graph_neighbors == 10


def test_ablation_configs_select_explicit_modes() -> None:
    fixed = load_config(ROOT / "configs" / "ablation-fixed-hybrid.yaml")
    no_vgae = load_config(ROOT / "configs" / "ablation-no-vgae.yaml")
    hard = load_config(ROOT / "configs" / "ablation-hard-gmm.yaml")
    assert fixed.fedhydra.hybrid_mode == "fixed"
    assert no_vgae.fedhydra.embedding_mode == "spectral"
    assert hard.fedhydra.hard_memberships
