"""The no-VGAE ablation feeds client summaries directly to the GMM."""

from pathlib import Path

import numpy as np
import torch

from fedsoar.config import load_config
from fedsoar.federated.server import FedSOARServer

ROOT = Path(__file__).resolve().parents[1]


def test_no_vgae_clusters_raw_node_features_and_reuses_cache(monkeypatch) -> None:
    config = load_config(
        ROOT / "configs" / "smoke.yaml",
        [
            "fedsoar.embedding_mode=summaries",
            "fedsoar.embedding_interval=2",
            "fedsoar.clustering_interval=2",
        ],
    ).fedsoar
    histograms = np.array([[0.8, 0.2], [0.75, 0.25], [0.2, 0.8], [0.25, 0.75]])
    rff_means = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    server = FedSOARServer(config, 4, 4, torch.device("cpu"), seed=0)

    def forbidden_embedding(*args, **kwargs):
        raise AssertionError("the direct-summary ablation must bypass graph embedding")

    monkeypatch.setattr("fedsoar.federated.server.fit_vgae", forbidden_embedding)
    monkeypatch.setattr("fedsoar.federated.server.spectral_embedding", forbidden_embedding)
    captured = []
    fit = server.gmm.fit

    def capture_gmm_input(features):
        captured.append(features.copy())
        return fit(features)

    monkeypatch.setattr(server.gmm, "fit", capture_gmm_input)
    diagnostics = server.refresh(0, histograms, rff_means)
    expected = np.concatenate([histograms, rff_means], axis=1)
    np.testing.assert_array_equal(captured[0], expected)
    np.testing.assert_array_equal(server.embeddings, expected)
    assert diagnostics.embedding_updated and diagnostics.clustering_updated
    assert diagnostics.vgae_loss is None

    cached = server.responsibilities.copy()
    diagnostics = server.refresh(1, histograms, rff_means * 2)
    np.testing.assert_array_equal(server.embeddings, expected)
    np.testing.assert_array_equal(server.responsibilities, cached)
    assert not diagnostics.embedding_updated and not diagnostics.clustering_updated
