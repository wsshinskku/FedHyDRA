"""Image datasets and client-specific feature-domain transforms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from fedhydra.config import DataConfig

TensorTransform = Callable[[Any], torch.Tensor]


@dataclass(slots=True)
class DatasetBundle:
    train: Dataset
    test: Dataset
    train_targets: np.ndarray
    num_classes: int
    class_names: list[str]
    train_transform: TensorTransform
    summary_transform: TensorTransform
    test_transform: TensorTransform
    channels: int = 3


class SyntheticImages(Dataset):
    """Small deterministic classification dataset used by tests and CI."""

    def __init__(
        self,
        samples: int,
        classes: int,
        image_size: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        targets = torch.arange(samples, dtype=torch.long) % classes
        targets = targets[torch.randperm(samples, generator=generator)]
        images = torch.randn(samples, 3, image_size, image_size, generator=generator) * 0.08
        for index, label in enumerate(targets.tolist()):
            channel = label % 3
            row = (label * 3) % max(image_size - 3, 1)
            column = (label * 5) % max(image_size - 3, 1)
            images[index, channel, row : row + 3, column : column + 3] += 1.5
            images[index] += label / max(classes - 1, 1) * 0.1
        self.images = images
        self.targets = targets.tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index].clone(), int(self.targets[index])


class TinyImageNetValidation(Dataset):
    """Read the original flat Tiny-ImageNet validation layout without moving files."""

    def __init__(self, root: Path, class_to_idx: dict[str, int]) -> None:
        annotation = root / "val" / "val_annotations.txt"
        image_root = root / "val" / "images"
        if not annotation.is_file():
            raise FileNotFoundError(f"Tiny-ImageNet annotations not found: {annotation}")
        entries: list[tuple[Path, int]] = []
        with annotation.open("r", encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2 or fields[1] not in class_to_idx:
                    continue
                entries.append((image_root / fields[0], class_to_idx[fields[1]]))
        self.entries = entries
        self.targets = [target for _, target in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, target = self.entries[index]
        with Image.open(path) as image:
            return image.convert("RGB"), target


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform: TensorTransform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.base[index]
        return self.transform(image), int(target)


def apply_domain_shift(
    image: torch.Tensor,
    domain_weights: Sequence[float] | None,
    strength: float,
) -> torch.Tensor:
    """Apply a deterministic, label-preserving color shift for structured clients.

    This is an explicit benchmark completion because the manuscript describes
    feature-level factors but does not specify an executable image transform.
    A boundary client's transform is the convex combination of group transforms.
    """

    if strength <= 0 or domain_weights is None or image.ndim != 3 or image.shape[0] < 3:
        return image
    weights = np.asarray(domain_weights, dtype=np.float64)
    if weights.sum() <= 0:
        return image
    weights = weights / weights.sum()
    gains = torch.ones(3, dtype=image.dtype, device=image.device)
    biases = torch.zeros(3, dtype=image.dtype, device=image.device)
    for domain, weight in enumerate(weights.tolist()):
        angle = 2.0 * np.pi * domain / max(len(weights), 1)
        pattern = torch.tensor(
            [np.cos(angle), np.cos(angle + 2.0), np.cos(angle + 4.0)],
            dtype=image.dtype,
            device=image.device,
        )
        gains += float(weight * strength) * 0.35 * pattern
        biases += float(weight * strength) * 0.10 * torch.roll(pattern, shifts=1)
    output = image.clone()
    output[:3] = output[:3] * gains[:, None, None] + biases[:, None, None]
    return output


class ClientDataset(Dataset):
    """A view over one client's indices, transform, and persistent domain mixture."""

    def __init__(
        self,
        base: Dataset,
        indices: Sequence[int],
        transform: TensorTransform,
        domain_weights: Sequence[float] | None = None,
        domain_shift_strength: float = 0.0,
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.domain_weights = domain_weights
        self.domain_shift_strength = domain_shift_strength

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        image, target = self.base[int(self.indices[position])]
        image = self.transform(image)
        image = apply_domain_shift(image, self.domain_weights, self.domain_shift_strength)
        return image, int(target)


def _identity_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.float()
    return transforms.functional.to_tensor(value)


def _image_transforms(name: str, image_size: int) -> tuple[TensorTransform, TensorTransform]:
    if name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return train, evaluation
    if name == "tiny_imagenet":
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        evaluation = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train, evaluation
    if name == "stl10":
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2566, 0.2713)
        train = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return train, evaluation
    return _identity_tensor, _identity_tensor


def build_datasets(config: DataConfig, seed: int) -> DatasetBundle:
    """Load raw datasets while keeping client and summary transforms separate."""

    train_transform, evaluation_transform = _image_transforms(config.name, config.image_size)
    root = Path(config.root).expanduser()

    if config.name == "synthetic":
        classes = max(config.structured_groups * 2, 4)
        budget = config.samples_per_client or 20
        train = SyntheticImages(config.num_clients * budget, classes, config.image_size, seed)
        test = SyntheticImages(max(classes * 12, 48), classes, config.image_size, seed + 1)
        return DatasetBundle(
            train=train,
            test=TransformDataset(test, evaluation_transform),
            train_targets=np.asarray(train.targets, dtype=np.int64),
            num_classes=classes,
            class_names=[f"class_{index}" for index in range(classes)],
            train_transform=train_transform,
            summary_transform=evaluation_transform,
            test_transform=evaluation_transform,
        )

    if config.name == "cifar100":
        train = datasets.CIFAR100(root, train=True, transform=None, download=config.download)
        test_raw = datasets.CIFAR100(root, train=False, transform=None, download=config.download)
        class_names = list(train.classes)
    elif config.name == "stl10":
        train = datasets.STL10(root, split="train", transform=None, download=config.download)
        test_raw = datasets.STL10(root, split="test", transform=None, download=config.download)
        class_names = list(train.classes)
    elif config.name == "tiny_imagenet":
        train_root = root / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(
                f"Tiny-ImageNet training directory not found at {train_root}. "
                "Run scripts/download_tiny_imagenet.py or set data.root."
            )
        train = datasets.ImageFolder(train_root, transform=None)
        test_raw = TinyImageNetValidation(root, train.class_to_idx)
        class_names = list(train.classes)
    else:
        raise ValueError(f"unsupported dataset: {config.name}")

    targets = getattr(train, "targets", getattr(train, "labels", None))
    if targets is None:
        raise TypeError(f"dataset {config.name} does not expose targets")
    return DatasetBundle(
        train=train,
        test=TransformDataset(test_raw, evaluation_transform),
        train_targets=np.asarray(targets, dtype=np.int64),
        num_classes=len(class_names),
        class_names=class_names,
        train_transform=train_transform,
        summary_transform=evaluation_transform,
        test_transform=evaluation_transform,
    )

