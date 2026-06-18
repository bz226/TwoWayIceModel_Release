"""Models for the Euler-1 reduced-data surrogate benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceBaseline(nn.Module):
    """Return the known Euler-1 field unchanged."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0:1]


class ResidualBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size, padding=padding),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResNetCNNSurrogate(nn.Module):
    """Moderate local convolutional baseline for Euler-1 prediction."""

    def __init__(self, input_channels: int = 7, width: int = 32, num_blocks: int = 4):
        super().__init__()
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.num_blocks = int(num_blocks)

        blocks: list[nn.Module] = [
            nn.Conv2d(self.input_channels, self.width, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        blocks.extend(ResidualBlock(self.width, kernel_size=3) for _ in range(self.num_blocks))
        blocks.extend(
            [
                nn.Conv2d(self.width, self.width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.width, 1, kernel_size=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SpectralConv2d(nn.Module):
    """2D Fourier layer with active low modes in x and y."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weights_pos = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights_neg = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul2d(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)
        max_modes1 = min(self.modes1, height)
        max_modes2 = min(self.modes2, width // 2 + 1)

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :max_modes1, :max_modes2] = self.compl_mul2d(
            x_ft[:, :, :max_modes1, :max_modes2],
            self.weights_pos[:, :, :max_modes1, :max_modes2],
        )
        out_ft[:, :, -max_modes1:, :max_modes2] = self.compl_mul2d(
            x_ft[:, :, -max_modes1:, :max_modes2],
            self.weights_neg[:, :, :max_modes1, :max_modes2],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class FNO2dSurrogate(nn.Module):
    """Euler-1 FNO with explicit input channels and four active spectral blocks."""

    def __init__(self, input_channels: int = 7, modes: int = 12, width: int = 16, num_blocks: int = 4):
        super().__init__()
        if num_blocks != 4:
            raise ValueError("FNO2dSurrogate is intended to use exactly 4 active spectral blocks")
        self.input_channels = int(input_channels)
        self.modes = int(modes)
        self.width = int(width)
        self.num_blocks = int(num_blocks)

        self.lift = nn.Conv2d(self.input_channels, self.width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(self.width, self.modes) for _ in range(self.num_blocks))
        self.project = nn.Sequential(
            nn.Conv2d(self.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.lift(features)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


def build_model(model_name: str, **kwargs) -> nn.Module:
    name = model_name.lower()
    if name == "persistence":
        return PersistenceBaseline()
    if name == "resnet_cnn":
        return ResNetCNNSurrogate(
            input_channels=kwargs.get("input_channels", 7),
            width=kwargs.get("cnn_width", 32),
            num_blocks=kwargs.get("cnn_blocks", 4),
        )
    if name == "fno":
        return FNO2dSurrogate(
            input_channels=kwargs.get("input_channels", 7),
            modes=kwargs.get("fno_modes", 12),
            width=kwargs.get("fno_width", 16),
            num_blocks=4,
        )
    raise ValueError("Unknown model. Choose from: persistence, resnet_cnn, fno")
