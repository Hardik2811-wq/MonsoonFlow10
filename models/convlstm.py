"""2-Layer ConvLSTM for Spatiotemporal Dynamics in MonsoonFlow-10.
Retains spatial memory, wind-moisture transport, and pressure advection.
Layer 0 uses 3x3 spatial gates (advection radius=1), Layer 1 uses 1x1 gates
for temporal mixing, keeping total Day-1 shock spread strictly localized.
"""
from typing import Tuple, List
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell with spatial convolutions inside gates."""
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """2-layer ConvLSTM processing sequential spatial feature representations."""
    def __init__(self, in_channels: int = 64, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        cells = []
        for l in range(num_layers):
            c_in = in_channels if l == 0 else hidden_dim
            k_size = 3 if l == 0 else 1
            cells.append(ConvLSTMCell(c_in, hidden_dim, kernel_size=k_size))
        self.cells = nn.ModuleList(cells)

    def init_hidden(self, batch_size: int, height: int = 32, width: int = 36, device: torch.device = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            states.append((h, c))
        return states

    def step(
        self,
        x: torch.Tensor,
        states: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process a single time-step across all ConvLSTM layers."""
        new_states = []
        h = x
        for l in range(self.num_layers):
            h, c = self.cells[l](h, states[l])
            new_states.append((h, c))
        return h, new_states
