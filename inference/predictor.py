"""Prediction engine with cyclic DOY rotation, MF10 10-step rollout, and zero-shock parity."""
from __future__ import annotations
import math
import time
from typing import Any, List, Optional, Tuple
import numpy as np

from .loader import registry
from .schemas import decode_array, NUM_CHANNELS, NUM_LATS, NUM_LONS, INPUT_DAYS

# Constants for cyclic DOY advancement without datetime libraries
RAD_PER_DAY = 2.0 * math.pi / 365.0


def is_ready() -> bool:
    return registry.is_ready


def get_model_name() -> str:
    return registry.model_name


def rotate_doy_sincos(sin_val: float, cos_val: float, days_delta: int = 1) -> Tuple[float, float]:
    """Rotate [sin, cos] by `days_delta` days using pure trigonometric angle addition.
    Guarantees zero datetime/date imports or literals in AST.
    """
    if days_delta == 0:
        return sin_val, cos_val
    angle = days_delta * RAD_PER_DAY
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    new_sin = sin_val * cos_a + cos_val * sin_a
    new_cos = cos_val * cos_a - sin_val * sin_a
    norm = math.hypot(new_sin, new_cos)
    if norm > 1e-9:
        return new_sin / norm, new_cos / norm
    return new_sin, new_cos


def simulate(
    init_states_raw: Any,
    doy_sincos: List[float],
    steps: int,
    perturbation_raw: Optional[Any] = None,
) -> tuple[np.ndarray, float]:
    """Execute 10-step autoregressive rollout in physical units under strict zero-shock contract.

    Args:
        init_states_raw: (3, 6, 32, 36) in physical units (t2m in K, tp in mm/day, etc.)
        doy_sincos: [sin, cos] of day t_0 (last input slice)
        steps: Number of forecast steps (default 10)
        perturbation_raw: Optional (6, 32, 36) physical delta shock or None

    Returns:
        out: (steps, 6, 32, 36) physical forecast array
        inference_ms: Execution duration in milliseconds
    """
    t0 = time.perf_counter()

    # 1. Decode initial conditions (3, 6, 32, 36) in physical units
    history = decode_array(init_states_raw, expect_shape=(INPUT_DAYS, NUM_CHANNELS, NUM_LATS, NUM_LONS)).copy()

    # 2. Perturbation handling & Zero-shock contract
    # Add physical delta P directly to init_states[-1] BEFORE normalization.
    # If perturbation is null or all zeros, bypass addition entirely for bit-level parity.
    if perturbation_raw is not None:
        pert = decode_array(perturbation_raw, expect_shape=(NUM_CHANNELS, NUM_LATS, NUM_LONS))
        if np.any(np.abs(pert) > 1e-12):
            history[-1] = history[-1] + pert
            history[-1, 1] = np.maximum(history[-1, 1], 0.0)
            history[-1, 0] = np.clip(history[-1, 0], 233.15, 333.15)
            history[-1, 5] = np.maximum(history[-1, 5], 0.0)

    # 3. Extract preloaded normalization tensors
    mean = registry.mean_np if registry.mean_np is not None else np.zeros((1, 6, 1, 1), dtype=np.float32)
    std = registry.std_np if registry.std_np is not None else np.ones((1, 6, 1, 1), dtype=np.float32)
    static = registry.static_terrain_np  # (2, 32, 36)

    # 4. Run MonsoonFlow-10 Autoregressive Rollout
    neural_model = registry.model if (registry.model is not None and callable(registry.model)) else None

    if neural_model is not None:
        import torch

        # Normalize dynamic channels: (3, 6, 32, 36)
        norm_history = (history - mean) / np.maximum(std, 1e-6)

        # Build cyclic DOY planes for t_{-2}, t_{-1}, t_0
        s0, c0 = float(doy_sincos[0]), float(doy_sincos[1])
        doy_planes = np.zeros((INPUT_DAYS, 2, NUM_LATS, NUM_LONS), dtype=np.float32)
        for idx, offset in enumerate((-2, -1, 0)):
            si, ci = rotate_doy_sincos(s0, c0, days_delta=offset)
            doy_planes[idx, 0] = si
            doy_planes[idx, 1] = ci

        if static is not None:
            static_tiled = np.broadcast_to(static[None, ...], (INPUT_DAYS, 2, NUM_LATS, NUM_LONS))
        else:
            static_tiled = np.zeros((INPUT_DAYS, 2, NUM_LATS, NUM_LONS), dtype=np.float32)

        # Concatenate into 10-channel input: (1, 3, 10, 32, 36)
        model_in = np.concatenate([norm_history, static_tiled, doy_planes], axis=1)
        in_tensor = torch.from_numpy(model_in).unsqueeze(0)

        with torch.no_grad():
            pred_norm = neural_model(in_tensor, steps=steps, perturbation=None)
            pred_norm_np = pred_norm.squeeze(0).cpu().numpy()  # (steps, 6, 32, 36)

        # De-normalize back to physical units
        out = (pred_norm_np * std + mean).astype(np.float32)
    else:
        # Fallback persistence if model checkpoint is absent
        last = history[-1].copy()
        last[1] = np.maximum(last[1], 0.0)
        last[0] = np.clip(last[0], 233.15, 333.15)
        out = np.repeat(last[None, ...], steps, axis=0).astype(np.float32)

    # 5. Enforce final physical boundaries on all rollout steps
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
    out[:, 5] = np.maximum(out[:, 5], 0.0)

    inference_ms = (time.perf_counter() - t0) * 1000.0
    return out, inference_ms
