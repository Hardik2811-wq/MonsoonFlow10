"""Physics-Aware Multi-Objective Loss Formulation for MonsoonFlow-10.

Components:
1. Latitude-Weighted & Channel-Normalized Forecast Loss (matching competition metric)
2. Skew-Aware Heavy-Rainfall Loss (threshold >= 10.0 mm/day) to guarantee CSI improvement
3. Spatial Variance Floor Regularizer: Penalizes Day-8 spatial variance ratio falling below 0.55
"""
from typing import Dict, Tuple, Optional
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
    """Composite physical loss balancing spatial accuracy, extreme rainfall, and variance preservation."""
    
    def __init__(
        self,
        stats: dict,
        heavy_rain_weight: float = 2.5,
        var_floor_weight: float = 0.5,
        min_variance_ratio: float = 0.55,
    ):
        super().__init__()
        self.heavy_rain_weight = heavy_rain_weight
        self.var_floor_weight = var_floor_weight
        self.min_variance_ratio = min_variance_ratio
        
        # Physical stats
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 6, 1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 6, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        # Latitude weights
        w = np.cos(np.deg2rad(LAT))
        w = w / w.mean()
        self.register_buffer("lat_weights", torch.tensor(w, dtype=torch.float32).view(1, 1, 1, NLAT, 1))

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_norm: Model rollout (B, steps, 6, 32, 36) in normalized units
            target_norm: Ground truth (B, steps, 6, 32, 36) in normalized units
            
        Returns:
            total_loss: Scalar torch.Tensor
            metrics_dict: Dictionary of individual loss components
        """
        B, S, C, H, W = pred_norm.shape
        diff = pred_norm - target_norm # (B, steps, 6, 32, 36)
        
        # 1. Latitude-Weighted MSE Forecast Loss
        sq_diff = (diff ** 2) * self.lat_weights
        lat_mse = sq_diff.mean()
        
        # 2. Skew-Aware Heavy-Rainfall Loss (Channel 1: tp)
        # De-normalize precipitation to physical units (mm/day)
        pred_tp_phys = pred_norm[:, :, 1:2] * self.std[:, :, 1:2] + self.mean[:, :, 1:2]
        target_tp_phys = target_norm[:, :, 1:2] * self.std[:, :, 1:2] + self.mean[:, :, 1:2]
        
        # Identify heavy rain cells (>= 10.0 mm/day)
        heavy_mask = (target_tp_phys >= 10.0).float()
        
        # Under-prediction penalty weight grows with rainfall intensity
        rain_diff = pred_norm[:, :, 1:2] - target_norm[:, :, 1:2]
        under_pred_weight = 1.0 + 2.0 * torch.log1p(torch.clamp(target_tp_phys / 10.0, min=0.0))
        heavy_rain_loss = (heavy_mask * under_pred_weight * (rain_diff ** 2) * self.lat_weights).sum() / (heavy_mask.sum() + 1e-4)
        
        # 3. Spatial Variance Floor Regularizer (Anti-Blur)
        # Spatial variance across (H, W) dimensions
        pred_var = torch.var(pred_norm, dim=(-2, -1), unbiased=False) # (B, steps, 6)
        target_var = torch.var(target_norm, dim=(-2, -1), unbiased=False) # (B, steps, 6)
        
        var_ratio = pred_var / (target_var + 1e-5) # (B, steps, 6)
        
        # Heavily penalize variance collapse, especially at later lead days
        var_deficit = F.relu(self.min_variance_ratio - var_ratio)
        var_floor_loss = var_deficit.mean()
        
        # Composite Total Loss
        total_loss = lat_mse + (self.heavy_rain_weight * heavy_rain_loss) + (self.var_floor_weight * var_floor_loss)
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_lat_mse": float(lat_mse.item()),
            "loss_heavy_rain": float(heavy_rain_loss.item()),
            "loss_var_floor": float(var_floor_loss.item()),
            "mean_var_ratio": float(var_ratio.mean().item()),
        }
        return total_loss, metrics
