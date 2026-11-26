"""Residual Decoder for MonsoonFlow-10.
Decodes ConvLSTM latent representation (64 channels) to predicted normalized state change Delta X_t.
Input: (B, 64, 32, 36)
Output: (B, 6, 32, 36)
"""
import torch
import torch.nn as nn
from .encoder import ResBlock


class ResidualDecoder(nn.Module):
    """Decodes latent spatiotemporal dynamics to state increment Delta X_t."""
    def __init__(self, in_channels: int = 64, out_channels: int = 6, num_groups: int = 4):
        super().__init__()
        self.res1 = ResBlock(in_channels, num_groups=num_groups)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=True),
        )
        
        # Initialize output conv weights close to zero for stable residual delta learning
        nn.init.zeros_(self.out_conv[-1].bias)
        self.out_conv[-1].weight.data.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 64, 32, 36)
        Returns:
            delta_x: Predicted state change (B, 6, 32, 36)
        """
        h = self.res1(x)
        delta_x = self.out_conv(h)
        return delta_x
