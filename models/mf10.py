"""MonsoonFlow-10 (MF10) Spatiotemporal Climate Digital Twin.

Phase 4: Climatology-anchored anomaly formulation.

    x_k = C(doy_k) + a_k
    a_k = T( r * a_{k-1} ; u_{k-1}, v_{k-1} ) + dA_k

- C: training-split (2005-2018) smoothed daily climatology, stored as a buffer
  (normalized units, 365 rows indexed by day-of-year derived from doy_sincos).
- r: learned per-channel anomaly damping in (0.5, 0.995) -> long leads relax to
  climatology instead of drifting.
- T: bounded wind-driven 3x3 upwind advection-diffusion (models/transport.py),
  carries shocks downstream so they propagate outward without teleporting.
- dA_k: neural increment from Encoder -> 2-layer ConvLSTM -> Decoder. The
  encoder sees the anomaly field (not the raw state) + static + doy planes.
- Physical constraints: t2m in [233.15, 333.15] K, tp >= 0, tcwv >= 0.
- Exact zero-shock bypass when perturbation is None or all-zeros.

I/O signature unchanged: (B, T_in, 10, 32, 36) normalized -> (B, steps, 6, 32, 36).
"""
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from .encoder import SpatialEncoder
from .convlstm import ConvLSTM
from .decoder import ResidualDecoder
from .transport import AnomalyTransport

NLAT, NLON = 32, 36
DAYS_PER_YEAR = 365
# Index of Feb 29 in the 366-row leap-indexed climatology table
LEAP_DAY_ROW = 31 + 28
# Per-channel anomaly damping r in [lo, hi] (t2m, tp, u10, v10, msl, tcwv).
# Thermodynamic fields (t2m, tcwv) carry ocean memory -> floor 0.92 so shocks persist
# long enough to be transported outward; rain/winds/pressure decorrelate faster -> free.
DAMP_LO = (0.92, 0.5, 0.5, 0.5, 0.5, 0.92)
DAMP_HI = 0.995


def climatology_to_yday_table(clim366: np.ndarray, stats: dict) -> np.ndarray:
    """Convert a (366, 6, H, W) leap-indexed physical climatology into a (365, 6, H, W)
    normalized table where row p is non-leap day-of-year p+1 (matches doy_sincos)."""
    rows = [p if p < LEAP_DAY_ROW else p + 1 for p in range(DAYS_PER_YEAR)]
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.maximum(np.array(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1), 1e-6)
    return ((clim366[rows].astype(np.float32) - mean) / std).astype(np.float32)


class MonsoonFlow10(nn.Module):
    """MonsoonFlow-10 (MF10) Spatiotemporal Autoregressive Forecasting Engine."""

    def __init__(
        self,
        in_channels: int = 10,
        dynamic_channels: int = 6,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        stats: Optional[dict] = None,
        climatology: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_channels = dynamic_channels
        self.hidden_dim = hidden_dim

        self.encoder = SpatialEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.convlstm = ConvLSTM(in_channels=hidden_dim, hidden_dim=hidden_dim, num_layers=num_lstm_layers)
        self.decoder = ResidualDecoder(in_channels=hidden_dim, out_channels=dynamic_channels)
        self.transport = AnomalyTransport(channels=dynamic_channels)

        # Per-channel anomaly damping r = lo + (hi - lo) * sigmoid(logit)
        self.register_buffer("damp_lo", torch.tensor(DAMP_LO, dtype=torch.float32).view(1, dynamic_channels, 1, 1))
        self.damp_hi = DAMP_HI
        self.damp_logit = nn.Parameter(torch.full((1, dynamic_channels, 1, 1), 0.4))

        if stats is not None:
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 6, 1, 1)
            std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 6, 1, 1)
        else:
            mean = torch.zeros(1, 6, 1, 1)
            std = torch.ones(1, 6, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        if climatology is not None and stats is not None:
            clim = torch.from_numpy(climatology_to_yday_table(climatology, stats))
        else:
            clim = torch.zeros(DAYS_PER_YEAR, dynamic_channels, NLAT, NLON)
        self.register_buffer("clim_norm", clim)

        self.tp_min_norm = float(-self.mean[0, 1, 0, 0] / torch.clamp(self.std[0, 1, 0, 0], min=1e-6))
        self.tcwv_min_norm = float(-self.mean[0, 5, 0, 0] / torch.clamp(self.std[0, 5, 0, 0], min=1e-6))
        self.t2m_min_norm = float((233.15 - self.mean[0, 0, 0, 0]) / torch.clamp(self.std[0, 0, 0, 0], min=1e-6))
        self.t2m_max_norm = float((333.15 - self.mean[0, 0, 0, 0]) / torch.clamp(self.std[0, 0, 0, 0], min=1e-6))

        d_theta = 2.0 * np.pi / DAYS_PER_YEAR
        self.cos_step = float(np.cos(d_theta))
        self.sin_step = float(np.sin(d_theta))

    # ---------------------------------------------------------------- helpers
    def damping(self) -> torch.Tensor:
        return self.damp_lo + (self.damp_hi - self.damp_lo) * torch.sigmoid(self.damp_logit)

    def rotate_doy(self, doy_sin: torch.Tensor, doy_cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_sin = doy_sin * self.cos_step + doy_cos * self.sin_step
        next_cos = doy_cos * self.cos_step - doy_sin * self.sin_step
        return next_sin, next_cos

    def clim_at(self, doy_sin: torch.Tensor, doy_cos: torch.Tensor) -> torch.Tensor:
        """(B,) sin/cos -> (B, 6, H, W) normalized climatology for that day of year."""
        theta = torch.atan2(doy_sin, doy_cos)
        yday = torch.round(theta * DAYS_PER_YEAR / (2.0 * np.pi)).long() % DAYS_PER_YEAR
        row = (yday - 1) % DAYS_PER_YEAR
        return self.clim_norm[row]

    def wind_ms(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = x_norm[:, 2:3] * self.std[:, 2:3] + self.mean[:, 2:3]
        v = x_norm[:, 3:4] * self.std[:, 3:4] + self.mean[:, 3:4]
        return u, v

    def apply_physical_constraints(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Enforce strict physical laws without introducing Softplus positive drizzle bias."""
        t2m = torch.clamp(x_norm[:, 0:1], min=self.t2m_min_norm, max=self.t2m_max_norm)
        tp = torch.clamp(x_norm[:, 1:2], min=self.tp_min_norm)
        u10 = x_norm[:, 2:3]
        v10 = x_norm[:, 3:4]
        msl = x_norm[:, 4:5]
        tcwv = torch.clamp(x_norm[:, 5:6], min=self.tcwv_min_norm)
        return torch.cat([t2m, tp, u10, v10, msl, tcwv], dim=1)

    def _encode(self, weather: torch.Tensor, clim: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([weather - clim, aux], dim=1))

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        init_states: torch.Tensor,
        steps: int = 10,
        perturbation: Optional[torch.Tensor] = None,
        return_increments: bool = False,
    ):
        B, T_in, C, H, W = init_states.shape
        device = init_states.device

        lstm_states = self.convlstm.init_hidden(B, height=H, width=W, device=device)

        for t in range(T_in - 1):
            st = init_states[:, t]
            clim_t = self.clim_at(st[:, 8, 0, 0], st[:, 9, 0, 0])
            h_enc = self._encode(st[:, 0:6], clim_t, st[:, 6:10])
            _, lstm_states = self.convlstm.step(h_enc, lstm_states)

        # Exact zero-shock bypass: all-zero perturbation is identical to perturbation=None
        if perturbation is not None and torch.any(perturbation != 0):
            perturbed_weather = self.apply_physical_constraints(init_states[:, -1, 0:6] + perturbation)
            curr_step_t0 = torch.cat([perturbed_weather, init_states[:, -1, 6:]], dim=1)
        else:
            curr_step_t0 = init_states[:, -1]

        curr_weather = curr_step_t0[:, 0:6]
        static_fields = curr_step_t0[:, 6:8]
        doy_sin = curr_step_t0[:, 8, 0, 0]
        doy_cos = curr_step_t0[:, 9, 0, 0]
        curr_clim = self.clim_at(doy_sin, doy_cos)

        h_enc = self._encode(curr_weather, curr_clim, curr_step_t0[:, 6:10])
        h_lstm, lstm_states = self.convlstm.step(h_enc, lstm_states)

        r = self.damping()
        rollout_list = []
        increments = []

        for k in range(steps):
            doy_sin, doy_cos = self.rotate_doy(doy_sin, doy_cos)
            next_clim = self.clim_at(doy_sin, doy_cos)

            anomaly = curr_weather - curr_clim
            u_ms, v_ms = self.wind_ms(curr_weather)
            carried = self.transport(r * anomaly, u_ms, v_ms)
            delta_a = self.decoder(h_lstm)
            increments.append(delta_a)
            next_weather = self.apply_physical_constraints(next_clim + carried + delta_a)
            rollout_list.append(next_weather)

            if k < steps - 1:
                doy_sin_plane = doy_sin.view(B, 1, 1, 1).expand(B, 1, H, W)
                doy_cos_plane = doy_cos.view(B, 1, 1, 1).expand(B, 1, H, W)
                aux = torch.cat([static_fields, doy_sin_plane, doy_cos_plane], dim=1)
                h_enc = self._encode(next_weather, next_clim, aux)
                h_lstm, lstm_states = self.convlstm.step(h_enc, lstm_states)
                curr_weather = next_weather
                curr_clim = next_clim

        rollout = torch.stack(rollout_list, dim=1)
        if return_increments:
            return rollout, torch.stack(increments, dim=1)
        return rollout
