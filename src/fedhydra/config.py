"""Typed configuration loading with recursive YAML inheritance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "fedhydra"
    output_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    deterministic: bool = True


@dataclass(slots=True)
class DataConfig:
    name: str = "cifar100"
    root: str = "data"
    download: bool = True
    num_clients: int = 200
    clients_per_round: int = 20
    partition: str = "structured"
    alpha: float = 0.3
    samples_per_client: int | None = None
    min_samples_per_client: int = 16
    num_workers: int = 4
    image_size: int = 32
    structured_groups: int = 5
    overlap_ratios: list[float] = field(
        default_factory=lambda: [0.20, 0.20, 0.30, 0.10]
    )
    boundary_fraction: float = 0.20
    profile_concentration: float = 100.0
    domain_shift_strength: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    name: str = "mobilenet_v2_small"
    width_multiplier: float = 0.5
    dropout: float = 0.2


@dataclass(slots=True)
class FederatedConfig:
    rounds: int = 300
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1.0e-4
    server_learning_rate: float = 1.0
    method: str = "fedhydra"
    proximal_mu: float = 0.0
    eval_every: int = 1
    checkpoint_every: int = 20
    bootstrap_summaries: bool = True
    summary_batch_size: int = 128
    max_summary_samples: int | None = None
    amp: bool = True


@dataclass(slots=True)
class FedHyDRAConfig:
    rff_dimension: int = 256
    rff_gamma: float = 1.0
    normalize_features: bool = False
    hybrid_mode: str = "adaptive"
    embedding_dimension: int = 16
    vgae_hidden_dimension: int = 64
    vgae_epochs: int = 50
    vgae_learning_rate: float = 0.01
    vgae_kl_weight: float = 1.0
    vgae_positive_weighting: bool = False
    vgae_self_loops: bool = True
    embedding_mode: str = "vgae"
    gmm_clusters: int = 5
    gmm_max_iterations: int = 100
    gmm_reg_covar: float = 1.0e-6
    hard_memberships: bool = False
    graph_neighbors: int = 20
    histogram_beta: float = 1.0
    graph_temperature: float = 0.5
    hybrid_regularization: float = 0.01
    hybrid_learning_rate: float = 0.05
    hybrid_interval: int = 5
    embedding_interval: int = 10
    clustering_interval: int = 20
    scale_epsilon: float = 1.0e-12
    aggregation_epsilon: float = 1.0e-12
    sample_weighted_updates: bool = False
    mixture_share_scope: str = "all_clients"


@dataclass(slots=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    fedhydra: FedHyDRAConfig = field(default_factory=FedHyDRAConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        d, f, h = self.data, self.federated, self.fedhydra
        if d.num_clients < 2:
            raise ValueError("data.num_clients must be at least 2")
        if not 1 <= d.clients_per_round <= d.num_clients:
            raise ValueError("data.clients_per_round must be in [1, data.num_clients]")
        if d.partition not in {"iid", "dirichlet", "structured"}:
            raise ValueError("data.partition must be iid, dirichlet, or structured")
        if d.partition == "dirichlet" and d.alpha <= 0:
            raise ValueError("data.alpha must be positive")
        if d.min_samples_per_client < 1 or d.image_size < 1 or d.num_workers < 0:
            raise ValueError(
                "min_samples_per_client and image_size must be positive; num_workers cannot be negative"
            )
        if d.profile_concentration < 0 or d.domain_shift_strength < 0:
            raise ValueError("profile_concentration and domain_shift_strength cannot be negative")
        if not 0 <= d.boundary_fraction <= 1:
            raise ValueError("data.boundary_fraction must be in [0, 1]")
        if d.structured_groups < 2 or d.structured_groups > d.num_clients:
            raise ValueError("data.structured_groups must be in [2, data.num_clients]")
        if any(not 0 <= ratio < 1 for ratio in d.overlap_ratios):
            raise ValueError("all overlap ratios must be in [0, 1)")
        if f.rounds < 1 or f.local_epochs < 1 or f.batch_size < 1:
            raise ValueError("rounds, local_epochs, and batch_size must be positive")
        if f.learning_rate <= 0 or f.server_learning_rate <= 0:
            raise ValueError("local and server learning rates must be positive")
        if f.momentum < 0 or f.weight_decay < 0 or f.proximal_mu < 0:
            raise ValueError("momentum, weight decay, and proximal_mu cannot be negative")
        if f.method not in {"fedavg", "fedprox", "fedhydra"}:
            raise ValueError("federated.method must be fedavg, fedprox, or fedhydra")
        if f.method == "fedprox" and f.proximal_mu <= 0:
            raise ValueError("fedprox requires federated.proximal_mu > 0")
        if self.model.width_multiplier <= 0 or not 0 <= self.model.dropout < 1:
            raise ValueError("model width_multiplier must be positive and dropout must be in [0, 1)")
        if f.eval_every < 1 or f.checkpoint_every < 1:
            raise ValueError("evaluation and checkpoint intervals must be positive")
        positive = {
            "rff_dimension": h.rff_dimension,
            "embedding_dimension": h.embedding_dimension,
            "vgae_hidden_dimension": h.vgae_hidden_dimension,
            "vgae_epochs": h.vgae_epochs,
            "gmm_clusters": h.gmm_clusters,
            "graph_neighbors": h.graph_neighbors,
            "hybrid_interval": h.hybrid_interval,
            "embedding_interval": h.embedding_interval,
            "clustering_interval": h.clustering_interval,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"fedhydra.{name} must be positive")
        if h.gmm_clusters > d.num_clients:
            raise ValueError("gmm_clusters cannot exceed num_clients")
        if h.graph_neighbors >= d.num_clients:
            raise ValueError("graph_neighbors must be smaller than num_clients")
        if h.clustering_interval < h.embedding_interval:
            raise ValueError("clustering_interval must be >= embedding_interval")
        if h.histogram_beta <= 0 or h.graph_temperature <= 0:
            raise ValueError("histogram_beta and graph_temperature must be positive")
        if h.rff_gamma <= 0 or h.hybrid_learning_rate <= 0:
            raise ValueError("rff_gamma and hybrid_learning_rate must be positive")
        if h.hybrid_regularization <= 0 or h.scale_epsilon <= 0:
            raise ValueError("hybrid_regularization and scale_epsilon must be positive")
        if h.aggregation_epsilon < 0 or h.gmm_reg_covar < 0:
            raise ValueError("aggregation_epsilon and gmm_reg_covar cannot be negative")
        if h.gmm_max_iterations < 1:
            raise ValueError("gmm_max_iterations must be positive")
        if h.mixture_share_scope not in {"all_clients", "participants"}:
            raise ValueError(
                "fedhydra.mixture_share_scope must be all_clients or participants"
            )
        if h.hybrid_mode not in {"adaptive", "fixed", "jsd_only", "mmd_only"}:
            raise ValueError(
                "fedhydra.hybrid_mode must be adaptive, fixed, jsd_only, or mmd_only"
            )
        if h.embedding_mode not in {"vgae", "spectral"}:
            raise ValueError("fedhydra.embedding_mode must be vgae or spectral")
        if not 0 < h.vgae_learning_rate or not 0 <= h.vgae_kl_weight:
            raise ValueError("invalid VGAE optimization settings")


_SECTIONS = {
    "experiment": ExperimentConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "federated": FederatedConfig,
    "fedhydra": FedHyDRAConfig,
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic configuration inheritance at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    base_path = (path.parent / str(base_name)).resolve()
    return _deep_merge(_load_yaml_tree(base_path, seen), raw)


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if len(keys) < 2:
        raise ValueError(f"override must use section.key syntax: {dotted_key}")
    cursor = mapping
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested override below {key}")
        cursor = child
    cursor[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a config file and optional ``section.key=value`` overrides."""

    raw = _load_yaml_tree(Path(path))
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"override must be key=value: {expression}")
        key, encoded = expression.split("=", 1)
        _set_nested(raw, key, yaml.safe_load(encoded))

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise KeyError(f"unknown configuration sections: {sorted(unknown_sections)}")

    sections: dict[str, Any] = {}
    for name, section_type in _SECTIONS.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise TypeError(f"configuration section {name} must be a mapping")
        known = set(section_type.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise KeyError(f"unknown keys in {name}: {sorted(unknown)}")
        sections[name] = section_type(**values)

    config = Config(**sections)
    config.validate()
    return config
