"""Multi-Scale Residual CNN Encoder for MonsoonFlow-10.
Preserves the exact 32x36 spatial resolution without downsampling pooling.
Input: (B, in_channels=10, 32, 36)
Output: (B, out_channels=64, 32, 36)
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual Convolutional Block with GroupNorm and SiLU activation."""
    def __init__(self, channels: int, num_groups: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.act1 = nn.SiLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.act2 = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + res)


class SpatialEncoder(nn.Module):
    """Multi-scale spatial encoder processing 10 input channels to 64 latent channels."""
    def __init__(self, in_channels: int = 10, hidden_dim: int = 64, num_groups: int = 4):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.res1 = ResBlock(hidden_dim, num_groups=num_groups)
        self.res2 = ResBlock(hidden_dim, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 10, 32, 36)
        Returns:
            latent: Tensor of shape (B, 64, 32, 36)
        """
        h = self.in_conv(x)
        h = self.res1(h)
        h = self.res2(h)
        return h
