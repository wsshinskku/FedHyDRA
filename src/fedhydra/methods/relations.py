"""Equations (6)-(15): calibrated hybrid discrepancies and top-k graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RelationState:
    jsd: np.ndarray
    mmd2: np.ndarray
    jsd_scaled: np.ndarray
    mmd2_scaled: np.ndarray
    hybrid: np.ndarray
    similarity: np.ndarray
    directed_edges: np.ndarray
    adjacency: np.ndarray


def pairwise_jsd(histograms: np.ndarray) -> np.ndarray:
    histograms = np.asarray(histograms, dtype=np.float64)
    if histograms.ndim != 2:
        raise ValueError("histograms must be a two-dimensional array")
    if np.any(histograms <= 0):
        raise ValueError("JSD expects strictly positive smoothed histograms")
    histograms = histograms / histograms.sum(axis=1, keepdims=True)
    left = histograms[:, None, :]
    right = histograms[None, :, :]
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * (np.log(left) - np.log(midpoint)), axis=2)
    divergence += 0.5 * np.sum(right * (np.log(right) - np.log(midpoint)), axis=2)
    divergence = np.maximum(0.5 * (divergence + divergence.T), 0.0)
    np.fill_diagonal(divergence, 0.0)
    return divergence


def pairwise_squared_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.sum(values * values, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * (values @ values.T)
    distances = np.maximum(0.5 * (distances + distances.T), 0.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def calibrate(discrepancy: np.ndarray, epsilon: float) -> np.ndarray:
    clients = discrepancy.shape[0]
    if clients < 2:
        return np.zeros_like(discrepancy, dtype=np.float64)
    maximum = float(np.max(discrepancy[np.triu_indices(clients, k=1)]))
    scaled = discrepancy / (maximum + epsilon)
    np.fill_diagonal(scaled, 0.0)
    return scaled


def projected_hybrid_update(
    omega: float,
    jsd_scaled: np.ndarray,
    mmd2_scaled: np.ndarray,
    learning_rate: float,
    regularization: float,
) -> tuple[float, float]:
    upper = np.triu_indices(jsd_scaled.shape[0], k=1)
    mean_difference = float(np.mean(jsd_scaled[upper] - mmd2_scaled[upper]))
    gradient = mean_difference - regularization * (1.0 - 2.0 * omega)
    updated = float(np.clip(omega - learning_rate * gradient, 0.0, 1.0))
    return updated, gradient


def topk_adjacency(hybrid: np.ndarray, neighbors: int, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clients = hybrid.shape[0]
    if hybrid.shape != (clients, clients):
        raise ValueError("hybrid discrepancy must be square")
    if not 1 <= neighbors < clients:
        raise ValueError("neighbors must be in [1, clients-1]")
    similarity = np.exp(-hybrid / temperature)
    np.fill_diagonal(similarity, 0.0)
    ranking_values = similarity.copy()
    np.fill_diagonal(ranking_values, -np.inf)
    order = np.argsort(-ranking_values, axis=1, kind="stable")[:, :neighbors]
    directed = np.zeros((clients, clients), dtype=bool)
    directed[np.arange(clients)[:, None], order] = True
    support = directed | directed.T
    adjacency = np.where(support, similarity, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    return similarity, directed, adjacency


class RelationEngine:
    def __init__(
        self,
        clients: int,
        neighbors: int,
        temperature: float,
        scale_epsilon: float,
        omega: float = 0.5,
    ) -> None:
        self.clients = clients
        self.neighbors = min(neighbors, clients - 1)
        self.temperature = temperature
        self.scale_epsilon = scale_epsilon
        self.omega = float(omega)
        self._jsd_scaled: np.ndarray | None = None
        self._mmd2_scaled: np.ndarray | None = None

    def discrepancies(self, histograms: np.ndarray, rff_means: np.ndarray) -> None:
        if histograms.shape[0] != self.clients or rff_means.shape[0] != self.clients:
            raise ValueError("summary cache does not contain the configured client count")
        self.jsd = pairwise_jsd(histograms)
        self.mmd2 = pairwise_squared_distance(rff_means)
        self._jsd_scaled = calibrate(self.jsd, self.scale_epsilon)
        self._mmd2_scaled = calibrate(self.mmd2, self.scale_epsilon)

    def update_omega(self, learning_rate: float, regularization: float) -> float:
        if self._jsd_scaled is None or self._mmd2_scaled is None:
            raise RuntimeError("compute discrepancies before updating omega")
        self.omega, gradient = projected_hybrid_update(
            self.omega,
            self._jsd_scaled,
            self._mmd2_scaled,
            learning_rate,
            regularization,
        )
        return gradient

    def graph(self) -> RelationState:
        if self._jsd_scaled is None or self._mmd2_scaled is None:
            raise RuntimeError("compute discrepancies before building a graph")
        hybrid = self.omega * self._jsd_scaled + (1.0 - self.omega) * self._mmd2_scaled
        similarity, directed, adjacency = topk_adjacency(
            hybrid, self.neighbors, self.temperature
        )
        return RelationState(
            jsd=self.jsd,
            mmd2=self.mmd2,
            jsd_scaled=self._jsd_scaled,
            mmd2_scaled=self._mmd2_scaled,
            hybrid=hybrid,
            similarity=similarity,
            directed_edges=directed,
            adjacency=adjacency,
        )

