import numpy as np
import torch

from fedhydra.methods.vgae import (
    VariationalGraphAutoencoder,
    fit_vgae,
    spectral_embedding,
    vgae_loss,
)
from fedhydra.models.mobilenet import MobileNetV2Small


def test_mobilenet_exposes_penultimate_features() -> None:
    model = MobileNetV2Small(num_classes=6, width_multiplier=0.25, dropout=0.0)
    inputs = torch.randn(2, 3, 16, 16)
    features = model.extract_features(inputs)
    logits = model(inputs)
    assert features.shape == (2, model.feature_dimension)
    assert logits.shape == (2, 6)


def test_vgae_returns_deterministic_posterior_mean_shape() -> None:
    torch.manual_seed(2)
    features = np.eye(4, dtype=np.float32)
    adjacency = np.array(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        dtype=np.float32,
    )
    model = VariationalGraphAutoencoder(4, 6, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    embeddings, diagnostics = fit_vgae(
        model,
        optimizer,
        features,
        adjacency,
        epochs=3,
        kl_weight=0.01,
        device=torch.device("cpu"),
    )
    assert embeddings.shape == (4, 3)
    assert np.all(np.isfinite(embeddings))
    assert diagnostics.loss > 0


def test_spectral_ablation_has_requested_shape_and_is_finite() -> None:
    adjacency = np.array(
        [[0, 1, 0], [1, 0, 0.2], [0, 0.2, 0]], dtype=np.float64
    )
    embeddings = spectral_embedding(adjacency, dimension=5)
    assert embeddings.shape == (3, 5)
    assert np.all(np.isfinite(embeddings))


def test_vgae_elbo_terms_use_the_same_node_normalization() -> None:
    clients, dimensions = 4, 3
    latent = torch.zeros(clients, dimensions)
    mean = torch.ones(clients, dimensions)
    log_std = torch.zeros_like(mean)
    support = torch.zeros(clients, clients, dtype=torch.bool)
    loss, reconstruction, kl = vgae_loss(
        latent,
        mean,
        log_std,
        support,
        kl_weight=1.0,
        positive_weighting=False,
    )
    expected_reconstruction = (clients - 1) * np.log(2.0)
    expected_kl = 0.5 * dimensions
    assert np.isclose(float(reconstruction), expected_reconstruction)
    assert np.isclose(float(kl), expected_kl)
    assert torch.allclose(loss, reconstruction + kl)
