"""Laplace-smoothed label histograms and shared RBF random Fourier features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


class RandomFourierFeatures:
    """Fixed RFF approximation to ``exp(-gamma * ||x-y||^2)``."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        gamma: float,
        seed: int,
    ) -> None:
        if input_dimension < 1 or output_dimension < 1 or gamma <= 0:
            raise ValueError("RFF dimensions and gamma must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.projection = (
            torch.randn(input_dimension, output_dimension, generator=generator)
            * np.sqrt(2.0 * gamma)
        )
        self.phase = torch.rand(output_dimension, generator=generator) * (2.0 * np.pi)
        self.scale = float(np.sqrt(2.0 / output_dimension))

    @property
    def output_dimension(self) -> int:
        return int(self.phase.numel())

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        projection = self.projection.to(device=features.device, dtype=features.dtype)
        phase = self.phase.to(device=features.device, dtype=features.dtype)
        return self.scale * torch.cos(features @ projection + phase)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "projection": self.projection.clone(),
            "phase": self.phase.clone(),
            "scale": self.scale,
        }


@dataclass(slots=True)
class ClientSummary:
    label_histogram: np.ndarray
    rff_mean: np.ndarray
    sample_count: int


@torch.inference_mode()
def compute_client_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    rff: RandomFourierFeatures,
    num_classes: int,
    beta: float,
    device: torch.device,
    normalize_features: bool = True,
    max_samples: int | None = None,
) -> ClientSummary:
    if beta <= 0:
        raise ValueError("histogram smoothing beta must be positive")
    model.eval()
    counts = torch.zeros(num_classes, dtype=torch.float64)
    rff_sum = torch.zeros(rff.output_dimension, dtype=torch.float64)
    seen = 0
    for images, labels in loader:
        if max_samples is not None:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            images, labels = images[:remaining], labels[:remaining]
        images = images.to(device, non_blocking=True)
        features = model.extract_features(images)
        if normalize_features:
            features = functional.normalize(features, dim=1)
        mapped = rff.transform(features).double().cpu()
        labels = labels.long().cpu()
        counts += torch.bincount(labels, minlength=num_classes).double()
        rff_sum += mapped.sum(dim=0)
        seen += int(labels.numel())
    if seen == 0:
        raise ValueError("cannot summarize an empty client dataset")
    histogram = (counts + beta) / (seen + num_classes * beta)
    return ClientSummary(
        label_histogram=histogram.numpy(),
        rff_mean=(rff_sum / seen).numpy(),
        sample_count=seen,
    )

