"""Physics-Aware Multi-Objective Loss Formulation for MonsoonFlow-10.

Components:
1. Latitude-Weighted & Channel-Normalized Forecast Loss (balanced L2 + L1)
2. Balanced Heavy-Rainfall Loss (>= 10.0 mm/day) + Log1p Rain Matching (prevents tp spikes)
3. Two-Sided Spatial Variance Band Regularizer: Keeps Day-8 spatial variance ratio in [0.60, 1.60]
"""
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.preprocessing import NLAT, NLON

LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)


def get_latitude_weights(device: torch.device) -> torch.Tensor:
    """Return normalized cosine latitude weights: shape (1, 1, 32, 1)."""
    w = np.cos(np.deg2rad(LAT))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device).view(1, 1, NLAT, 1)


class PhysicsAwareLoss(nn.Module):
    """Composite physical loss balancing spatial accuracy, extreme rainfall, and variance bounds."""
    
    def __init__(
        self,
        stats: dict,
        heavy_rain_weight: float = 0.5,
        var_band_weight: float = 0.4,
        min_variance_ratio: float = 0.60,
        max_variance_ratio: float = 1.60,
        l1_weight: float = 0.3,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.heavy_rain_weight = heavy_rain_weight
        self.var_band_weight = var_band_weight
        self.min_variance_ratio = min_variance_ratio
        self.max_variance_ratio = max_variance_ratio
        
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 6, 1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 6, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        w = np.cos(np.deg2rad(LAT))
        w = w / w.mean()
        self.register_buffer("lat_weights", torch.tensor(w, dtype=torch.float32).view(1, 1, 1, NLAT, 1))

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, S, C, H, W = pred_norm.shape
        diff = pred_norm - target_norm
        
        # 1. Latitude-Weighted MSE Forecast Loss across all 6 channels
        sq_diff = (diff ** 2) * self.lat_weights
        lat_mse = sq_diff.mean()
        lat_l1 = (diff.abs() * self.lat_weights).mean()
        
        # 2. Balanced Heavy-Rainfall & Log-Rain Loss (Channel 1: tp)
        pred_tp_phys = torch.clamp(pred_norm[:, :, 1:2] * self.std[:, :, 1:2] + self.mean[:, :, 1:2], min=0.0)
        target_tp_phys = torch.clamp(target_norm[:, :, 1:2] * self.std[:, :, 1:2] + self.mean[:, :, 1:2], min=0.0)
        
        heavy_mask = (target_tp_phys >= 10.0).float()
        rain_diff = pred_norm[:, :, 1:2] - target_norm[:, :, 1:2]
        # Normalized by total grid cells (not heavy_mask.sum()) so gradients never explode
        heavy_rain_loss = (heavy_mask * (rain_diff ** 2) * self.lat_weights[:, :, 0:1]).mean() * 5.0
        log_rain_loss = (torch.log1p(pred_tp_phys) - torch.log1p(target_tp_phys)) ** 2
        log_rain_loss = (log_rain_loss * self.lat_weights[:, :, 0:1]).mean()
        
        # 3. Two-Sided Spatial Variance Band Regularizer [0.60, 1.60]
        pred_var = torch.var(pred_norm, dim=(-2, -1), unbiased=False)
        target_var = torch.var(target_norm, dim=(-2, -1), unbiased=False)
        var_ratio = pred_var / (target_var + 1e-4)
        
        lower_deficit = F.relu(self.min_variance_ratio - var_ratio)
        upper_excess = F.relu(var_ratio - self.max_variance_ratio)
        var_band_loss = (lower_deficit + upper_excess).mean()
        
        total_loss = (
            lat_mse
            + self.l1_weight * lat_l1
            + self.heavy_rain_weight * (heavy_rain_loss + 0.5 * log_rain_loss)
            + self.var_band_weight * var_band_loss
        )
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_lat_mse": float(lat_mse.item()),
            "loss_lat_l1": float(lat_l1.item()),
            "loss_heavy_rain": float(heavy_rain_loss.item()),
            "loss_var_band": float(var_band_loss.item()),
            "mean_var_ratio": float(var_ratio.mean().item()),
            "tp_var_ratio": float(var_ratio[:, -1, 1].mean().item()),
        }
        return total_loss, metrics
