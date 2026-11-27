"""
MAUSAM 2D Spectral & Open-Boundary Physics Audit (evaluation/eval_physics_audit.py).

Computes two publication-grade diagnostics on the frozen MonsoonFlow-10 checkpoint:
1. 2D Spatial Power-Spectrum Ratio (MAUSAM, Gupta et al. 2026):
   Radially averaged 2D FFT kinetic energy and precipitation (tp) power-spectrum
   ratios (Pred / True) at Days 1, 4, and 8 on Pool A (2019-2020) and 2021 holdout.
2. Open-Boundary Moisture Residual (R_q) & Non-Negativity Audit (Sha et al. 2025 / Chapman et al. 2026):
   Verifies 0% tp < 0 / tcwv <= 0 violations and compares MF10's 2D column moisture
   tendency + divergence + tp residual against ERA5's own 2D open-boundary residual envelope.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import MonsoonDataset  # noqa: E402
from data.preprocessing import chronological_split, load_raw_dataset  # noqa: E402
from models.backbones import load_model_from_checkpoint  # noqa: E402


LATS = np.linspace(6.5, 37.5, 32, dtype=np.float64)


LEAD_DAYS = (1, 4, 8)
METERS_PER_DEG = 111_320.0
SEC_PER_DAY = 86_400.0


def _radial_bins(h: int, w: int) -> tuple[np.ndarray, int]:
    """Return integer radial wavenumber grid (centered) and max valid ring."""
    ky = np.arange(h) - h // 2
    kx = np.arange(w) - w // 2
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    r = np.round(np.sqrt(yy**2 + xx**2)).astype(int)
    k_max = min(h, w) // 2
    return r, k_max


def radial_power_spectrum_2d(field: np.ndarray, r_grid: np.ndarray, k_max: int) -> np.ndarray:
    """Compute radially averaged 2D FFT power spectrum for k = 1..k_max."""
    centered = field - np.mean(field)
    fft_c = np.fft.fftshift(np.fft.fft2(centered))
    power_2d = (np.abs(fft_c) ** 2) / float(field.size)
    spec = np.zeros(k_max, dtype=np.float64)
    for k in range(1, k_max + 1):
        mask = r_grid == k
        spec[k - 1] = float(np.mean(power_2d[mask])) if np.any(mask) else 0.0
    return spec


def radial_ke_spectrum_2d(u10: np.ndarray, v10: np.ndarray, r_grid: np.ndarray, k_max: int) -> np.ndarray:
    """Compute radially averaged 2D FFT kinetic energy spectrum 0.5*(|U|^2 + |V|^2)."""
    pu = radial_power_spectrum_2d(u10, r_grid, k_max)
    pv = radial_power_spectrum_2d(v10, r_grid, k_max)
    return 0.5 * (pu + pv)


def compute_spectral_audit(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute Pred/True 2D power-spectrum ratios for KE and tp at Days 1, 4, 8."""
    w_count, _, _, h, w = y_pred.shape
    r_grid, k_max = _radial_bins(h, w)
    results = {}

    for day in LEAD_DAYS:
        idx = day - 1
        ke_pred_sum = np.zeros(k_max, dtype=np.float64)
        ke_true_sum = np.zeros(k_max, dtype=np.float64)
        tp_pred_sum = np.zeros(k_max, dtype=np.float64)
        tp_true_sum = np.zeros(k_max, dtype=np.float64)

        for n in range(w_count):
            ke_pred_sum += radial_ke_spectrum_2d(y_pred[n, idx, 2], y_pred[n, idx, 3], r_grid, k_max)
            ke_true_sum += radial_ke_spectrum_2d(y_true[n, idx, 2], y_true[n, idx, 3], r_grid, k_max)
            tp_pred_sum += radial_power_spectrum_2d(y_pred[n, idx, 1], r_grid, k_max)
            tp_true_sum += radial_power_spectrum_2d(y_true[n, idx, 1], r_grid, k_max)

        ke_ratio_k = ke_pred_sum / np.maximum(ke_true_sum, 1e-12)
        tp_ratio_k = tp_pred_sum / np.maximum(tp_true_sum, 1e-12)

        results[f"day_{day}"] = {
            "ke_total_ratio": float(np.sum(ke_pred_sum) / np.maximum(np.sum(ke_true_sum), 1e-12)),
            "ke_synoptic_k1_4_ratio": float(np.mean(ke_ratio_k[0:4])),
            "ke_mesoscale_k5_16_ratio": float(np.mean(ke_ratio_k[4:k_max])),
            "tp_total_ratio": float(np.sum(tp_pred_sum) / np.maximum(np.sum(tp_true_sum), 1e-12)),
            "tp_synoptic_k1_4_ratio": float(np.mean(tp_ratio_k[0:4])),
            "tp_mesoscale_k5_16_ratio": float(np.mean(tp_ratio_k[4:k_max])),
        }
    return results


def _moisture_residual_2d(rollout: np.ndarray) -> np.ndarray:
    """
    Compute open-boundary 2D column moisture residual R_q (kg/m^2/day):
        R_q = d(tcwv)/dt + div(tcwv * v10) + tp
    over interior grid cells [1:-1, 1:-1] for consecutive steps t -> t+1.
    """
    # rollout: (W, T, 6, 32, 36)
    cos_lat = np.cos(np.deg2rad(LATS[1:-1])).reshape(1, 1, -1, 1)
    dx = 2.0 * METERS_PER_DEG * cos_lat  # central diff over 2 deg lon
    dy = 2.0 * METERS_PER_DEG            # central diff over 2 deg lat

    tcwv = rollout[:, :, 5]  # (W, T, 32, 36)
    u10 = rollout[:, :, 2]
    v10 = rollout[:, :, 3]
    tp = rollout[:, :, 1]

    d_tcwv_dt = tcwv[:, 1:, 1:-1, 1:-1] - tcwv[:, :-1, 1:-1, 1:-1]
    tcwv_mid = 0.5 * (tcwv[:, :-1] + tcwv[:, 1:])
    u_mid = 0.5 * (u10[:, :-1] + u10[:, 1:])
    v_mid = 0.5 * (v10[:, :-1] + v10[:, 1:])
    tp_mid = 0.5 * (tp[:, :-1, 1:-1, 1:-1] + tp[:, 1:, 1:-1, 1:-1])

    fx = tcwv_mid * u_mid
    fy = tcwv_mid * v_mid
    dfx_dx = (fx[:, :, 1:-1, 2:] - fx[:, :, 1:-1, :-2]) / dx
    dfy_dy = (fy[:, :, 2:, 1:-1] - fy[:, :, :-2, 1:-1]) / dy
    div_flux = (dfx_dx + dfy_dy) * SEC_PER_DAY

    return d_tcwv_dt + div_flux + tp_mid


def compute_moisture_and_nonneg_audit(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Verify 0% non-negativity violations and compare MF10 R_q against ERA5."""
    tp_pred = y_pred[:, :, 1]
    tcwv_pred = y_pred[:, :, 5]
    total_cells = int(tp_pred.size)

    tp_neg_count = int(np.sum(tp_pred < 0.0))
    tcwv_nonpos_count = int(np.sum(tcwv_pred <= 0.0))

    rq_pred = _moisture_residual_2d(y_pred)
    rq_era5 = _moisture_residual_2d(y_true)

    abs_pred = np.abs(rq_pred)
    abs_era5 = np.abs(rq_era5)

    rms_pred = float(np.sqrt(np.mean(rq_pred**2)))
    rms_era5 = float(np.sqrt(np.mean(rq_era5**2)))
    mae_pred = float(np.mean(abs_pred))
    mae_era5 = float(np.mean(abs_era5))

    return {
        "non_negativity": {
            "total_cells": total_cells,
            "tp_lt_0_count": tp_neg_count,
            "tp_lt_0_pct": float(100.0 * tp_neg_count / total_cells),
            "tcwv_le_0_count": tcwv_nonpos_count,
            "tcwv_le_0_pct": float(100.0 * tcwv_nonpos_count / total_cells),
            "tp_min_mm_day": float(np.min(tp_pred)),
            "tcwv_min_kg_m2": float(np.min(tcwv_pred)),
            "all_finite": bool(np.all(np.isfinite(y_pred))),
        },
        "open_boundary_rq_kg_m2_day": {
            "mf10_mean_abs_rq": mae_pred,
            "era5_mean_abs_rq": mae_era5,
            "mean_abs_ratio_mf10_to_era5": float(mae_pred / max(mae_era5, 1e-12)),
            "mf10_rms_rq": rms_pred,
            "era5_rms_rq": rms_era5,
            "rms_ratio_mf10_to_era5": float(rms_pred / max(rms_era5, 1e-12)),
            "mf10_p95_abs_rq": float(np.percentile(abs_pred, 95)),
            "era5_p95_abs_rq": float(np.percentile(abs_era5, 95)),
            "mf10_mean_signed_rq": float(np.mean(rq_pred)),
            "era5_mean_signed_rq": float(np.mean(rq_era5)),
        },
    }


def run_split_audit(
    model: torch.nn.Module,
    splits: dict,
    stats: dict,
    static_fields: np.ndarray,
    split: str,
    num_windows: int = 50,
    is_backbone_ckpt: bool = True,
) -> dict:
    """Run frozen MF10 rollouts on `split` and compute spectral + R_q audits."""
    mean_np = np.array(stats["mean"], dtype=np.float32).reshape(1, 1, 6, 1, 1)
    std_np = np.array(stats["std"], dtype=np.float32).reshape(1, 1, 6, 1, 1)

    split_data, split_dates = splits[split]
    ds = MonsoonDataset(split_data, split_dates, stats, static_fields, t_in=3, t_rollout=10, step_stride=4)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    pred_list, true_list = [], []
    with torch.no_grad():
        for in_seq, tgt_seq, _ in loader:
            tgt_norm = tgt_seq.cpu().numpy()
            if is_backbone_ckpt:
                init_phys = in_seq[:, :, 0:6] * torch.from_numpy(std_np) + torch.from_numpy(mean_np)
                doy_last = in_seq[:, -1, 8:10, 0, 0]
                pred_phys = model(init_phys, doy_last, steps=10).cpu().numpy()
            else:
                pred_norm = model(in_seq, steps=10).cpu().numpy()
                pred_phys = pred_norm * std_np + mean_np
            pred_list.append(pred_phys)
            true_list.append(tgt_norm * std_np + mean_np)
            if len(pred_list) * 16 >= num_windows:
                break

    y_pred = np.concatenate(pred_list, axis=0)[:num_windows]
    y_true = np.concatenate(true_list, axis=0)[:num_windows]

    return {
        "split": split,
        "num_windows": int(len(y_pred)),
        "spectral_power_ratio": compute_spectral_audit(y_pred, y_true),
        "moisture_and_nonneg": compute_moisture_and_nonneg_audit(y_pred, y_true),
    }


def main() -> None:
    ckpt_path = REPO_ROOT / "checkpoints" / "model.pt"
    stats_path = REPO_ROOT / "checkpoints" / "stats.json"
    static_path = REPO_ROOT / "checkpoints" / "static.npy"
    nc_path = REPO_ROOT / "data" / "india_climate_train.nc"

    with open(stats_path) as f:
        stats = json.load(f)
    static_fields = np.load(static_path)

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_backbone_ckpt = isinstance(raw_ckpt, dict) and "model_config" in raw_ckpt
    if is_backbone_ckpt:
        from models.backbones import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(ckpt_path, map_location="cpu")
    else:
        model = MonsoonFlow10(in_channels=10, dynamic_channels=6, hidden_dim=64, num_lstm_layers=2, stats=stats)
        state = raw_ckpt["state_dict"] if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt else raw_ckpt
        model.load_state_dict(state)
        model.eval()

    raw_data, raw_dates = load_raw_dataset(nc_path)
    splits = chronological_split(raw_data, raw_dates)

    report = {
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        "pool_a_val": run_split_audit(model, splits, stats, static_fields, split="val", num_windows=50, is_backbone_ckpt=is_backbone_ckpt),
        "holdout_2021_test": run_split_audit(model, splits, stats, static_fields, split="test", num_windows=50, is_backbone_ckpt=is_backbone_ckpt),
    }

    out_path = REPO_ROOT / "evaluation" / "physics_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nPhysics audit saved to: {out_path}")


if __name__ == "__main__":
    main()
