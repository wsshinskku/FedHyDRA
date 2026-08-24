"""Full-covariance GMM soft memberships and WSS diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(slots=True)
class GMMState:
    responsibilities: np.ndarray
    mixture_weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    wss: float
    converged: bool
    iterations: int


class SoftGMM:
    def __init__(
        self,
        clusters: int,
        regularization: float,
        max_iterations: int,
        seed: int,
    ) -> None:
        self.estimator = GaussianMixture(
            n_components=clusters,
            covariance_type="full",
            reg_covar=regularization,
            max_iter=max_iterations,
            n_init=5,
            init_params="kmeans",
            random_state=seed,
            warm_start=False,
        )

    def fit(self, embeddings: np.ndarray) -> GMMState:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.shape[0] < self.estimator.n_components:
            raise ValueError("GMM component count exceeds the number of clients")
        self.estimator.fit(embeddings)
        responsibilities = self.estimator.predict_proba(embeddings)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        mixture_weights = responsibilities.mean(axis=0)
        differences = embeddings[:, None, :] - self.estimator.means_[None, :, :]
        squared = np.sum(differences * differences, axis=2)
        wss = float(np.sum(responsibilities * squared))
        return GMMState(
            responsibilities=responsibilities,
            mixture_weights=mixture_weights,
            means=self.estimator.means_.copy(),
            covariances=self.estimator.covariances_.copy(),
            wss=wss,
            converged=bool(self.estimator.converged_),
            iterations=int(self.estimator.n_iter_),
        )
