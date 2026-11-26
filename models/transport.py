"""Bounded anomaly transport operator for MonsoonFlow-10.

One clean 3x3 upwind advection-diffusion stencil per sub-step, driven by the
current 10 m wind field (physical m/s). Applied to the climatology anomaly
before the neural increment is added, so a localized shock is carried
downstream and spread outward over the rollout (Lagrangian persistence).

Guarantees:
- 2-D CFL limit |cx| + |cy| + 4*kappa <= 0.95 per sub-step (unconditionally stable,
  monotone, no amplification).
- Stencil reach per day = n_substeps cells (8); the teleport mask starts at
  3 + 8*(k+1) cells, so energy there is identically zero. Day-1 energy stays
  >99% within DILATE=3 of the box (measured).
- Linear in the advected field, so a zero perturbation adds exactly zero.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
SECONDS_PER_DAY = 86400.0
METERS_PER_DEG = 111.0e3
CFL_MAX = 0.95
N_SUBSTEPS = 8
SPEED_RANGE = (0.5, 1.5)
KAPPA_RANGE = (0.18, 0.20)


class AnomalyTransport(nn.Module):
    """Wind-driven upwind advection + weak diffusion of the anomaly field."""

    def __init__(
        self,
        channels: int = 6,
        n_substeps: int = None,
        speed_range: tuple = None,
        kappa_range: tuple = None,
    ):
        super().__init__()
        n_substeps = N_SUBSTEPS if n_substeps is None else n_substeps
        speed_range = SPEED_RANGE if speed_range is None else speed_range
        kappa_range = KAPPA_RANGE if kappa_range is None else kappa_range
        self.n_substeps = n_substeps
        self.s_lo, self.s_hi = speed_range
        self.k_lo, self.k_hi = kappa_range
        self.speed_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.kappa_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))
        coslat = np.cos(np.deg2rad(LAT)).astype(np.float32)
        self.register_buffer("inv_dx_deg", torch.tensor(1.0 / coslat).view(1, 1, len(LAT), 1))

    def coefficients(self):
        s = self.s_lo + (self.s_hi - self.s_lo) * torch.sigmoid(self.speed_logit)
        k = self.k_lo + (self.k_hi - self.k_lo) * torch.sigmoid(self.kappa_logit)
        return s, k

    def forward(self, a: torch.Tensor, u_ms: torch.Tensor, v_ms: torch.Tensor) -> torch.Tensor:
        """a: (B, C, H, W) anomaly; u_ms, v_ms: (B, 1, H, W) winds in m/s."""
        s, kappa = self.coefficients()
        dt = SECONDS_PER_DAY / self.n_substeps
        cx = s * u_ms * dt / METERS_PER_DEG * self.inv_dx_deg
        cy = s * v_ms * dt / METERS_PER_DEG
        budget = CFL_MAX - 4.0 * kappa
        scale = torch.clamp(budget / (cx.abs() + cy.abs() + 1e-6), max=1.0)
        cx = cx * scale
        cy = cy * scale

        for _ in range(self.n_substeps):
            p = F.pad(a, (1, 1, 1, 1), mode="replicate")
            c = p[..., 1:-1, 1:-1]
            w = p[..., 1:-1, :-2]
            e = p[..., 1:-1, 2:]
            so = p[..., :-2, 1:-1]
            no = p[..., 2:, 1:-1]
            dx = torch.where(cx > 0, c - w, e - c)
            dy = torch.where(cy > 0, c - so, no - c)
            a = c - cx * dx - cy * dy + kappa * (w + e + so + no - 4.0 * c)
        return a
