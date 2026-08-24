"""A self-contained lightweight MobileNetV2 with an explicit feature API."""

from __future__ import annotations

import torch
from torch import nn

from fedhydra.config import ModelConfig


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int, expand: int) -> None:
        super().__init__()
        hidden = input_channels * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(ConvNormActivation(input_channels, hidden, kernel_size=1))
        layers.extend(
            [
                ConvNormActivation(hidden, hidden, stride=stride, groups=hidden),
                nn.Conv2d(hidden, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and input_channels == output_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        return inputs + output if self.use_residual else output


class MobileNetV2Small(nn.Module):
    """MobileNetV2 scaled for 32-96 pixel research benchmarks."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        width_multiplier: float = 0.5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive")
        initial = _make_divisible(32 * width_multiplier)
        settings = [
            # expansion, output channels, repeats, first stride
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 3, 2),
            (6, 96, 2, 1),
            (6, 160, 2, 2),
            (6, 320, 1, 1),
        ]
        layers: list[nn.Module] = [ConvNormActivation(input_channels, initial, stride=1)]
        current = initial
        for expansion, channels, repeats, first_stride in settings:
            output = _make_divisible(channels * width_multiplier)
            for repeat in range(repeats):
                stride = first_stride if repeat == 0 else 1
                layers.append(InvertedResidual(current, output, stride, expansion))
                current = output
        final = _make_divisible(512 * max(1.0, width_multiplier))
        layers.append(ConvNormActivation(current, final, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.feature_dimension = final
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(final, num_classes))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return torch.mean(features, dim=(-2, -1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


class TinyConvNet(nn.Module):
    """Minimal feature-exposing network reserved for smoke tests and CI."""

    def __init__(self, num_classes: int, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.feature_dimension = 32
        self.classifier = nn.Linear(32, num_classes)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs).mean(dim=(-2, -1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


def build_model(
    config: ModelConfig,
    num_classes: int,
    input_channels: int = 3,
) -> nn.Module:
    if config.name == "mobilenet_v2_small":
        return MobileNetV2Small(
            num_classes=num_classes,
            input_channels=input_channels,
            width_multiplier=config.width_multiplier,
            dropout=config.dropout,
        )
    if config.name == "tiny_cnn":
        return TinyConvNet(num_classes=num_classes, input_channels=input_channels)
    raise ValueError(f"unsupported model: {config.name}")
