"""Multi-Scale Residual CNN Encoder for MonsoonFlow-10.
Preserves the exact 32x36 spatial resolution without downsampling pooling.
Uses pixel-wise ChannelLayerNorm (zero spatial leakage across HxW) and a
strictly bounded receptive field (radius=2) to guarantee Day-1 shock localization.
"""
import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """Per-pixel normalization strictly across the channel dimension (C).
    Unlike GroupNorm/BatchNorm, this never mixes statistics across (H, W),
    eliminating instantaneous spatial teleportation of localized shocks.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class ResBlock(nn.Module):
    """Residual Convolutional Block with ChannelLayerNorm and SiLU activation."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln1 = ChannelLayerNorm(channels)
        self.act1 = nn.SiLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.ln2 = ChannelLayerNorm(channels)
        self.act2 = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        return self.act2(out + res)


class SpatialEncoder(nn.Module):
    """Spatial encoder processing 10 input channels to 64 latent channels (receptive radius = 2)."""
    def __init__(self, in_channels: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            ChannelLayerNorm(48),
            nn.SiLU(inplace=False),
            nn.Conv2d(48, hidden_dim, kernel_size=1, padding=0, bias=False),
            ChannelLayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.res1 = ResBlock(hidden_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        h = self.res1(h)
        return h
