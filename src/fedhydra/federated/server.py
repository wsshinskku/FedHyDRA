"""Server-side FedHyDRA structural refresh and aggregation state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from fedhydra.config import FedHyDRAConfig
from fedhydra.methods.aggregation import aggregate_fedhydra
from fedhydra.methods.gmm import GMMState, SoftGMM
from fedhydra.methods.relations import RelationEngine, RelationState
from fedhydra.methods.vgae import (
    VariationalGraphAutoencoder,
    fit_vgae,
    spectral_embedding,
)


@dataclass(slots=True)
class RefreshDiagnostics:
    omega_updated: bool = False
    embedding_updated: bool = False
    clustering_updated: bool = False
    omega_gradient: float | None = None
    vgae_loss: float | None = None
    vgae_reconstruction: float | None = None
    vgae_kl: float | None = None
    gmm_wss: float | None = None
    gmm_converged: bool | None = None


class FedHyDRAServer:
    def __init__(
        self,
        config: FedHyDRAConfig,
        clients: int,
        node_feature_dimension: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.config = config
        self.clients = clients
        self.device = device
        initial_omega = {
            "adaptive": 0.5,
            "fixed": 0.5,
            "jsd_only": 1.0,
            "mmd_only": 0.0,
        }[config.hybrid_mode]
        self.relations = RelationEngine(
            clients=clients,
            neighbors=config.graph_neighbors,
            temperature=config.graph_temperature,
            scale_epsilon=config.scale_epsilon,
            omega=initial_omega,
        )
        self.vgae = VariationalGraphAutoencoder(
            node_feature_dimension,
            config.vgae_hidden_dimension,
            config.embedding_dimension,
        ).to(device)
        self.vgae_optimizer = torch.optim.Adam(
            self.vgae.parameters(), lr=config.vgae_learning_rate
        )
        self.gmm = SoftGMM(
            clusters=config.gmm_clusters,
            regularization=config.gmm_reg_covar,
            max_iterations=config.gmm_max_iterations,
            seed=seed,
        )
        self.embeddings = np.zeros(
            (clients, config.embedding_dimension), dtype=np.float64
        )
        self.responsibilities = np.full(
            (clients, config.gmm_clusters), 1.0 / config.gmm_clusters
        )
        self.mixture_weights = np.full(config.gmm_clusters, 1.0 / config.gmm_clusters)
        self.relation_state: RelationState | None = None
        self.gmm_state: GMMState | None = None

    @property
    def omega(self) -> float:
        return self.relations.omega

    @property
    def wss(self) -> float | None:
        return None if self.gmm_state is None else self.gmm_state.wss

    def refresh(
        self,
        round_index: int,
        histograms: np.ndarray,
        rff_means: np.ndarray,
    ) -> RefreshDiagnostics:
        cfg = self.config
        diagnostics = RefreshDiagnostics()
        weight_due = round_index % cfg.hybrid_interval == 0
        embedding_due = round_index % cfg.embedding_interval == 0
        clustering_due = round_index % cfg.clustering_interval == 0

        if (weight_due and cfg.hybrid_mode == "adaptive") or embedding_due:
            self.relations.discrepancies(histograms, rff_means)
        if weight_due and cfg.hybrid_mode == "adaptive":
            diagnostics.omega_gradient = self.relations.update_omega(
                cfg.hybrid_learning_rate, cfg.hybrid_regularization
            )
            diagnostics.omega_updated = True
        if embedding_due:
            self.relation_state = self.relations.graph()
            node_features = np.concatenate([histograms, rff_means], axis=1)
            if cfg.embedding_mode == "vgae":
                self.embeddings, vgae = fit_vgae(
                    self.vgae,
                    self.vgae_optimizer,
                    node_features,
                    self.relation_state.adjacency,
                    cfg.vgae_epochs,
                    cfg.vgae_kl_weight,
                    self.device,
                    positive_weighting=cfg.vgae_positive_weighting,
                    add_self_loops=cfg.vgae_self_loops,
                )
                diagnostics.vgae_loss = vgae.loss
                diagnostics.vgae_reconstruction = vgae.reconstruction
                diagnostics.vgae_kl = vgae.kl
            else:
                self.embeddings = spectral_embedding(
                    self.relation_state.adjacency, cfg.embedding_dimension
                )
            diagnostics.embedding_updated = True
        if clustering_due:
            self.gmm_state = self.gmm.fit(self.embeddings)
            if cfg.hard_memberships:
                responsibilities = np.eye(cfg.gmm_clusters)[
                    self.gmm_state.responsibilities.argmax(axis=1)
                ]
                differences = (
                    self.embeddings[:, None, :] - self.gmm_state.means[None, :, :]
                )
                hard_wss = float(
                    np.sum(responsibilities * np.sum(differences * differences, axis=2))
                )
                self.gmm_state = GMMState(
                    responsibilities=responsibilities,
                    mixture_weights=responsibilities.mean(axis=0),
                    means=self.gmm_state.means,
                    covariances=self.gmm_state.covariances,
                    wss=hard_wss,
                    converged=self.gmm_state.converged,
                    iterations=self.gmm_state.iterations,
                )
            self.responsibilities = self.gmm_state.responsibilities
            self.mixture_weights = self.gmm_state.mixture_weights
            diagnostics.clustering_updated = True
            diagnostics.gmm_wss = self.gmm_state.wss
            diagnostics.gmm_converged = self.gmm_state.converged
        return diagnostics

    def aggregate(
        self,
        updates: Sequence[dict[str, torch.Tensor]],
        participant_ids: Sequence[int],
        sample_counts: Sequence[int],
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        return aggregate_fedhydra(
            updates=updates,
            participant_ids=participant_ids,
            responsibilities=self.responsibilities,
            mixture_weights=self.mixture_weights,
            sample_counts=sample_counts,
            epsilon=self.config.aggregation_epsilon,
            sample_weighted=self.config.sample_weighted_updates,
            mixture_share_scope=self.config.mixture_share_scope,
        )

    def state_dict(self) -> dict[str, Any]:
        gmm_state = None
        if self.gmm_state is not None:
            gmm_state = asdict(self.gmm_state)
            gmm_state = {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in gmm_state.items()
            }
        return {
            "omega": self.omega,
            "vgae": self.vgae.state_dict(),
            "vgae_optimizer": self.vgae_optimizer.state_dict(),
            "embeddings": torch.from_numpy(self.embeddings),
            "responsibilities": torch.from_numpy(self.responsibilities),
            "mixture_weights": torch.from_numpy(self.mixture_weights),
            "gmm_state": gmm_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.relations.omega = float(state["omega"])
        self.vgae.load_state_dict(state["vgae"])
        self.vgae_optimizer.load_state_dict(state["vgae_optimizer"])
        self.embeddings = state["embeddings"].cpu().numpy()
        self.responsibilities = state["responsibilities"].cpu().numpy()
        self.mixture_weights = state["mixture_weights"].cpu().numpy()
        if state.get("gmm_state") is not None:
            gmm_state = {
                key: value.cpu().numpy() if torch.is_tensor(value) else value
                for key, value in state["gmm_state"].items()
            }
            self.gmm_state = GMMState(**gmm_state)
