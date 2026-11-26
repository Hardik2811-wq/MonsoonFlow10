"""Residual Decoder for MonsoonFlow-10.
Decodes ConvLSTM latent representation (64 channels) to predicted normalized state change Delta X_t.
Uses 1x1 channel-mixing convolutions and ChannelLayerNorm to avoid spatial shock bleed.
"""
import torch
import torch.nn as nn
from .encoder import ResBlock


class ResidualDecoder(nn.Module):
    """Decodes latent spatiotemporal dynamics to state increment Delta X_t."""
    def __init__(self, in_channels: int = 64, out_channels: int = 6):
        super().__init__()
        self.res1 = ResBlock(in_channels, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0, bias=False),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, out_channels, kernel_size=1, padding=0, bias=True),
        )
        
        # Exact zero init: the untrained model equals the climatology-anchored prior
        nn.init.zeros_(self.out_conv[-1].bias)
        nn.init.zeros_(self.out_conv[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res1(x)
        delta_x = self.out_conv(h)
        return delta_x
