"""Balanced classification accuracy and round-curve summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    loss: float
    samples: int


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> ClassificationMetrics:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    totals = torch.zeros(num_classes, dtype=torch.long)
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        predictions = logits.argmax(dim=1)
        labels_cpu = labels.cpu()
        matches = predictions.eq(labels).cpu()
        totals += torch.bincount(labels_cpu, minlength=num_classes)
        correct += torch.bincount(labels_cpu, weights=matches.float(), minlength=num_classes).long()
        total_samples += int(labels.numel())
    if total_samples == 0:
        raise ValueError("evaluation loader is empty")
    present = totals > 0
    per_class = correct[present].float() / totals[present].float()
    return ClassificationMetrics(
        accuracy=float(correct.sum().item() / total_samples),
        balanced_accuracy=float(per_class.mean().item()),
        loss=total_loss / total_samples,
        samples=total_samples,
    )


def summarize_curve(values: list[float], rounds: list[int], burn_in: int = 0) -> dict[str, float | int]:
    if len(values) != len(rounds) or not values:
        raise ValueError("values and rounds must be non-empty and aligned")
    curve = np.asarray(values, dtype=np.float64)
    round_array = np.asarray(rounds, dtype=np.int64)
    keep = round_array >= burn_in
    if not keep.any():
        raise ValueError("burn-in removes every evaluation point")
    window = curve[keep]
    mean = float(window.mean())
    final = float(curve[-1])
    threshold = 0.9 * final
    hits = np.flatnonzero(curve >= threshold)
    convergence = int(round_array[hits[0]]) if hits.size else int(round_array[-1])
    return {
        "final": final,
        "mean": mean,
        "cov": float(window.std(ddof=0) / mean) if mean else float("nan"),
        "min_mean_ratio": float(window.min() / mean) if mean else float("nan"),
        "convergence_round_90pct_final": convergence,
        "burn_in": burn_in,
    }

