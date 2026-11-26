"""MonsoonFlow-10 (MF10) Spatiotemporal Climate Digital Twin.

Architecture:
- Input per step: 10 channels (6 dynamic + 2 static + 2 doy_sincos rotated planes)
- Spatial Encoder: Multi-scale Residual CNN (GroupNorm, SiLU) preserving 32x36 resolution
- Temporal Dynamics Core: 2-layer 64-channel ConvLSTM
- Residual Decoder: Predicts state change Delta X_t (X_{t+1} = X_t + Delta X_t)
- Physical Constraints:
  * Softplus non-negativity for tp >= 0 and tcwv >= 0
  * Bounded clamping for t2m in [233.15, 333.15] K
- Cyclic DOY Rotation Matrix: Rotates [sin(theta), cos(theta)] forward at each lead day
"""
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpatialEncoder
from .convlstm import ConvLSTM
from .decoder import ResidualDecoder

NLAT, NLON = 32, 36


class MonsoonFlow10(nn.Module):
    """MonsoonFlow-10 (MF10) Spatiotemporal Autoregressive Forecasting Engine."""
    
    def __init__(
        self,
        in_channels: int = 10,
        dynamic_channels: int = 6,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        stats: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_channels = dynamic_channels
        self.hidden_dim = hidden_dim
        
        # Core neural sub-modules
        self.encoder = SpatialEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.convlstm = ConvLSTM(in_channels=hidden_dim, hidden_dim=hidden_dim, num_layers=num_lstm_layers)
        self.decoder = ResidualDecoder(in_channels=hidden_dim, out_channels=dynamic_channels)
        
        # Physical normalization constants (registered as buffers)
        if stats is not None:
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 6, 1, 1)
            std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 6, 1, 1)
        else:
            mean = torch.zeros(1, 6, 1, 1)
            std = torch.ones(1, 6, 1, 1)
            
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        # Precompute normalized bounds for physical enforcement
        self.tp_min_norm = float(-self.mean[0, 1, 0, 0] / torch.clamp(self.std[0, 1, 0, 0], min=1e-6))
        self.tcwv_min_norm = float(-self.mean[0, 5, 0, 0] / torch.clamp(self.std[0, 5, 0, 0], min=1e-6))
        self.t2m_min_norm = float((233.15 - self.mean[0, 0, 0, 0]) / torch.clamp(self.std[0, 0, 0, 0], min=1e-6))
        self.t2m_max_norm = float((333.15 - self.mean[0, 0, 0, 0]) / torch.clamp(self.std[0, 0, 0, 0], min=1e-6))

        # Precompute cyclic rotation step (2 * pi / 365)
        d_theta = 2.0 * np.pi / 365.0
        self.cos_step = float(np.cos(d_theta))
        self.sin_step = float(np.sin(d_theta))

    def rotate_doy(self, doy_sin: torch.Tensor, doy_cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate [sin(theta), cos(theta)] forward by one calendar day using 2D rotation matrix."""
        next_sin = doy_sin * self.cos_step + doy_cos * self.sin_step
        next_cos = doy_cos * self.cos_step - doy_sin * self.sin_step
        return next_sin, next_cos

    def apply_physical_constraints(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Enforce strict physical laws in normalized space without in-place tensor mutations."""
        t2m = torch.clamp(x_norm[:, 0:1], min=self.t2m_min_norm, max=self.t2m_max_norm)
        tp = F.softplus(x_norm[:, 1:2] - self.tp_min_norm, beta=4.0) + self.tp_min_norm
        u10 = x_norm[:, 2:3]
        v10 = x_norm[:, 3:4]
        msl = x_norm[:, 4:5]
        tcwv = F.softplus(x_norm[:, 5:6] - self.tcwv_min_norm, beta=4.0) + self.tcwv_min_norm
        return torch.cat([t2m, tp, u10, v10, msl, tcwv], dim=1)

    def forward(
        self,
        init_states: torch.Tensor,
        steps: int = 10,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout forecast.
        
        Args:
            init_states: Tensor of shape (B, T_in=3, 10, 32, 36)
            steps: Forecast horizon lead days (e.g., 1, 3, 5, 10)
            perturbation: Optional normalized shock added to t0 state (B, 6, 32, 36)
            
        Returns:
            rollout: Tensor of shape (B, steps, 6, 32, 36) normalized predictions
        """
        B, T_in, C, H, W = init_states.shape
        device = init_states.device
        
        # Initialize ConvLSTM hidden states
        lstm_states = self.convlstm.init_hidden(B, height=H, width=W, device=device)
        
        # Step 1: Burn-in historical sequence (t_{-2}, t_{-1})
        for t in range(T_in - 1):
            h_enc = self.encoder(init_states[:, t])
            _, lstm_states = self.convlstm.step(h_enc, lstm_states)
            
        # Step 2: Final input step (t_0) with optional perturbation
        if perturbation is not None:
            perturbed_weather = self.apply_physical_constraints(init_states[:, -1, 0:6] + perturbation)
            curr_step_t0 = torch.cat([perturbed_weather, init_states[:, -1, 6:]], dim=1)
        else:
            curr_step_t0 = init_states[:, -1]
            
        h_enc = self.encoder(curr_step_t0)
        h_lstm, lstm_states = self.convlstm.step(h_enc, lstm_states)
        
        # Current state and environmental anchors
        curr_weather = curr_step_t0[:, 0:6]
        static_fields = curr_step_t0[:, 6:8]
        
        # DOY sin/cos scalar values for t_0
        doy_sin = curr_step_t0[:, 8:9, 0, 0]
        doy_cos = curr_step_t0[:, 9:10, 0, 0]
        
        rollout_list = []
        
        # Step 3: Autoregressive forward rollout
        for k in range(steps):
            # Predict residual change Delta X_t
            delta_x = self.decoder(h_lstm)
            
            # Residual state update: X_{k} = X_{k-1} + Delta X_k
            next_weather = self.apply_physical_constraints(curr_weather + delta_x)
            rollout_list.append(next_weather)
            
            if k < steps - 1:
                # Rotate DOY forward for the next forecast day
                doy_sin, doy_cos = self.rotate_doy(doy_sin, doy_cos)
                doy_sin_plane = doy_sin.view(B, 1, 1, 1).expand(B, 1, H, W)
                doy_cos_plane = doy_cos.view(B, 1, 1, 1).expand(B, 1, H, W)
                
                # Assemble next step 10-channel input
                next_in = torch.cat([next_weather, static_fields, doy_sin_plane, doy_cos_plane], dim=1)
                
                # Forward through encoder and ConvLSTM
                h_enc = self.encoder(next_in)
                h_lstm, lstm_states = self.convlstm.step(h_enc, lstm_states)
                curr_weather = next_weather
                
        rollout = torch.stack(rollout_list, dim=1) # (B, steps, 6, 32, 36)
        return rollout
