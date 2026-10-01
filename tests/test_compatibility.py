"""Naming changes must preserve configurations, imports, and launch commands."""

from importlib import import_module
from pathlib import Path

import pytest

from fedhydra.cli import main as legacy_main
from fedhydra.config import FedHyDRAConfig
from fedsoar.cli import main
from fedsoar.config import FedSOARConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "config",
        "data.datasets",
        "data.partition",
        "evaluation.metrics",
        "federated.client",
        "federated.server",
        "federated.trainer",
        "methods.aggregation",
        "methods.gmm",
        "methods.relations",
        "methods.summaries",
        "methods.vgae",
        "models.mobilenet",
        "utils",
    ],
)
def test_import_aliases_share_module_identity(module: str) -> None:
    canonical = import_module(f"fedsoar.{module}")
    legacy = import_module(f"fedhydra.{module}")
    assert canonical is legacy
    parent, attribute = module.rsplit(".", 1) if "." in module else ("", module)
    package = import_module("fedsoar" + (f".{parent}" if parent else ""))
    assert getattr(package, attribute) is canonical


@pytest.mark.parametrize("base_key,child_key", [("fedsoar", "fedhydra"), ("fedhydra", "fedsoar")])
def test_aliases_normalize_before_config_inheritance(
    tmp_path: Path, base_key: str, child_key: str
) -> None:
    base = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    (tmp_path / "base.yaml").write_text(base.replace("fedsoar", base_key), encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        f"_base_: base.yaml\n{child_key}:\n  vgae_epochs: 7\n", encoding="utf-8"
    )
    config = load_config(child)
    assert config.fedsoar.vgae_epochs == 7
    assert config.fedsoar is config.fedhydra
    assert FedSOARConfig is FedHyDRAConfig
    assert config.federated.method == "fedsoar"
    assert "fedsoar" in config.to_dict() and "fedhydra" not in config.to_dict()

    overridden = load_config(
        child,
        ["fedsoar.vgae_epochs=8", "fedhydra.vgae_epochs=9", "federated.method=fedhydra"],
    )
    assert overridden.fedsoar.vgae_epochs == 9
    assert overridden.federated.method == "fedsoar"


def test_mixed_sections_in_same_file_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mixed.yaml"
    path.write_text("fedsoar: {}\nfedhydra: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only one"):
        load_config(path)


@pytest.mark.parametrize("entrypoint,program", [(main, "fedsoar"), (legacy_main, "fedhydra")])
def test_both_cli_entrypoints_are_available(entrypoint, program: str, capsys) -> None:
    with pytest.raises(SystemExit) as result:
        entrypoint(["--help"])
    assert result.value.code == 0
    assert f"usage: {program} " in capsys.readouterr().out
