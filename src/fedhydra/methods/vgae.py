"""Dense two-layer variational graph autoencoder for client graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


def normalize_adjacency(
    adjacency: torch.Tensor, add_self_loops: bool = True
) -> torch.Tensor:
    if add_self_loops:
        adjacency = adjacency + torch.eye(
            adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
        )
    degree = adjacency.sum(dim=1).clamp_min(1.0e-12)
    inverse = degree.rsqrt()
    return inverse[:, None] * adjacency * inverse[None, :]


class GraphConvolution(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dimension, output_dimension))
        self.bias = nn.Parameter(torch.zeros(output_dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ (features @ self.weight) + self.bias


class VariationalGraphAutoencoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, latent_dimension: int) -> None:
        super().__init__()
        self.shared = GraphConvolution(input_dimension, hidden_dimension)
        self.mean = GraphConvolution(hidden_dimension, latent_dimension)
        self.log_standard_deviation = GraphConvolution(hidden_dimension, latent_dimension)

    def encode(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = functional.relu(self.shared(features, adjacency))
        mean = self.mean(hidden, adjacency)
        log_std = self.log_standard_deviation(hidden, adjacency).clamp(-8.0, 8.0)
        return mean, log_std

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.encode(features, adjacency)
        latent = mean + torch.randn_like(mean) * torch.exp(log_std)
        return latent, mean, log_std


@dataclass(slots=True)
class VGAEDiagnostics:
    loss: float
    reconstruction: float
    kl: float


def spectral_embedding(weighted_adjacency: np.ndarray, dimension: int) -> np.ndarray:
    """Deterministic graph-only embedding for the explicit no-VGAE ablation."""

    adjacency = np.asarray(weighted_adjacency, dtype=np.float64)
    clients = adjacency.shape[0]
    with_loops = adjacency + np.eye(clients)
    degree = np.maximum(with_loops.sum(axis=1), 1.0e-12)
    normalized = degree[:, None] ** -0.5 * with_loops * degree[None, :] ** -0.5
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    retain = min(dimension, clients)
    indices = np.argsort(eigenvalues)[-retain:][::-1]
    embedding = eigenvectors[:, indices] * np.sqrt(np.abs(eigenvalues[indices]))[None, :]
    # Fix the arbitrary eigenvector sign for deterministic manifests and tests.
    for column in range(embedding.shape[1]):
        pivot = int(np.argmax(np.abs(embedding[:, column])))
        if embedding[pivot, column] < 0:
            embedding[:, column] *= -1
    if retain < dimension:
        embedding = np.pad(embedding, ((0, 0), (0, dimension - retain)))
    return embedding


def vgae_loss(
    latent: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    support: torch.Tensor,
    kl_weight: float,
    positive_weighting: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clients = latent.shape[0]
    mask = ~torch.eye(clients, dtype=torch.bool, device=latent.device)
    logits = latent @ latent.T
    targets = support.float()
    positive_weight = None
    if positive_weighting:
        positives = targets[mask].sum()
        negatives = mask.sum() - positives
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp_min(1.0)
    reconstruction = functional.binary_cross_entropy_with_logits(
        logits[mask],
        targets[mask],
        pos_weight=positive_weight,
        reduction="sum",
    ) / clients
    # Reconstruction and KL are both summed ELBO terms normalized by the same
    # node count. Averaging BCE over N(N-1) pairs while averaging KL over N
    # nodes would accidentally multiply the relative KL weight by N-1.
    kl = -0.5 * torch.sum(
        1.0 + 2.0 * log_std - mean.square() - torch.exp(2.0 * log_std)
    ) / clients
    return reconstruction + kl_weight * kl, reconstruction, kl


def fit_vgae(
    model: VariationalGraphAutoencoder,
    optimizer: torch.optim.Optimizer,
    node_features: np.ndarray,
    weighted_adjacency: np.ndarray,
    epochs: int,
    kl_weight: float,
    device: torch.device,
    positive_weighting: bool = False,
    add_self_loops: bool = True,
) -> tuple[np.ndarray, VGAEDiagnostics]:
    features = torch.as_tensor(node_features, dtype=torch.float32, device=device)
    adjacency = torch.as_tensor(weighted_adjacency, dtype=torch.float32, device=device)
    normalized = normalize_adjacency(adjacency, add_self_loops=add_self_loops)
    support = adjacency > 0
    model.train()
    diagnostics: VGAEDiagnostics | None = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        latent, mean, log_std = model(features, normalized)
        loss, reconstruction, kl = vgae_loss(
            latent,
            mean,
            log_std,
            support,
            kl_weight,
            positive_weighting=positive_weighting,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        diagnostics = VGAEDiagnostics(
            loss=float(loss.detach()),
            reconstruction=float(reconstruction.detach()),
            kl=float(kl.detach()),
        )
    model.eval()
    with torch.inference_mode():
        mean, _ = model.encode(features, normalized)
    if diagnostics is None:
        raise ValueError("VGAE epochs must be positive")
    return mean.cpu().numpy(), diagnostics
