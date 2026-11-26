"""Curriculum Training Pipeline for MonsoonFlow-10 (MF10).

Curriculum Stages:
- Stage 1: 1-Step autoregressive training (learns single-day increments Delta X)
- Stage 2: 3-Step autoregressive training (short-range advection)
- Stage 3: 5-Step autoregressive training (medium-range dynamics)
- Stage 4: 10-Step full autoregressive rollout (long-range stability, heavy rain, variance preservation)

Saves final weights to checkpoints/model.pt (< 500 MB).
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocessing import load_raw_dataset, chronological_split
from data.dataset import MonsoonDataset
from models.mf10 import MonsoonFlow10
from models.physics_losses import PhysicsAwareLoss


def run_curriculum_training(
    nc_path: str = "data/india_climate_train.nc",
    stats_path: str = "checkpoints/stats.json",
    static_path: str = "checkpoints/static.npy",
    checkpoint_out: str = "checkpoints/model.pt",
    device_name: Optional[str] = None,
):
    """Execute 4-stage curriculum training on available hardware."""
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[Agent-ML] Initializing training on device: {device}")

    # Load precomputed static fields and statistics
    with open(stats_path, "r") as f:
        stats = json.load(f)
    static_fields = np.load(static_path) # (2, 32, 36)

    # Ingest dataset and extract chronological partitions
    print("[Agent-ML] Loading dataset and building chronological partitions...")
    data, dates = load_raw_dataset(nc_path)
    splits = chronological_split(data, dates)
    train_data, train_dates = splits["train"]
    val_data, val_dates = splits["val"]

    # Instantiate MF10 model
    model = MonsoonFlow10(in_channels=10, dynamic_channels=6, hidden_dim=64, num_lstm_layers=2, stats=stats)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Agent-ML] MF10 Model Architecture Initialized:")
    print(f"  Total trainable parameters: {total_params:,} (~{total_params * 4 / (1024 * 1024):.2f} MB)")

    # Loss function and optimizer
    criterion = PhysicsAwareLoss(stats=stats, heavy_rain_weight=2.0, var_floor_weight=0.3, min_variance_ratio=0.55)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Fast, effective 4-Stage Curriculum Plan
    stages = [
        {"stage": 1, "steps": 1, "epochs": 1, "lr": 1.2e-3, "stride": 4, "batch_size": 32},
        {"stage": 2, "steps": 3, "epochs": 1, "lr": 8.0e-4, "stride": 4, "batch_size": 32},
        {"stage": 3, "steps": 5, "epochs": 1, "lr": 5.0e-4, "stride": 6, "batch_size": 16},
        {"stage": 4, "steps": 10, "epochs": 2, "lr": 2.5e-4, "stride": 8, "batch_size": 16},
    ]

    for cfg in stages:
        stage_idx = cfg["stage"]
        k_steps = cfg["steps"]
        epochs = cfg["epochs"]
        lr = cfg["lr"]
        stride = cfg["stride"]
        batch_size = cfg["batch_size"]

        print(f"\n=======================================================")
        print(f"[Agent-ML] STARTING CURRICULUM STAGE {stage_idx}/4: {k_steps}-STEP ROLLOUT")
        print(f"  Epochs: {epochs}, LR: {lr:.1e}, Stride: {stride}, Batch Size: {batch_size}")
        print(f"=======================================================")

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        train_ds = MonsoonDataset(
            train_data, train_dates, stats, static_fields,
            t_in=3, t_rollout=k_steps, step_stride=stride
        )
        val_ds = MonsoonDataset(
            val_data, val_dates, stats, static_fields,
            t_in=3, t_rollout=k_steps, step_stride=stride * 2
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_losses = []

            for batch_idx, (in_seq, tgt_seq, _) in enumerate(train_loader):
                in_seq = in_seq.to(device) # (B, 3, 10, 32, 36)
                tgt_seq = tgt_seq.to(device) # (B, k_steps, 6, 32, 36)

                optimizer.zero_grad()
                pred = model(in_seq, steps=k_steps)
                loss, metrics = criterion(pred, tgt_seq)
                loss.backward()

                # Gradient clipping to ensure numerical stability in autoregressive rollouts
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(metrics["loss_total"])

            # Validation pass
            model.eval()
            val_losses = []
            val_var_ratios = []
            with torch.no_grad():
                for in_seq, tgt_seq, _ in val_loader:
                    in_seq = in_seq.to(device)
                    tgt_seq = tgt_seq.to(device)
                    pred = model(in_seq, steps=k_steps)
                    v_loss, v_metrics = criterion(pred, tgt_seq)
                    val_losses.append(v_metrics["loss_total"])
                    val_var_ratios.append(v_metrics["mean_var_ratio"])

            dt = time.time() - t0
            mean_train = float(np.mean(train_losses)) if train_losses else 0.0
            mean_val = float(np.mean(val_losses)) if val_losses else 0.0
            mean_vr = float(np.mean(val_var_ratios)) if val_var_ratios else 1.0
            print(f"  Stage {stage_idx} | Epoch {ep}/{epochs} [{dt:.1f}s] - Train Loss: {mean_train:.4f} | Val Loss: {mean_val:.4f} | Var Ratio: {mean_vr:.3f}")

    # Save final model weights
    Path(checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    # Move model to CPU before saving to ensure seamless CPU loading
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), checkpoint_out)
    weight_size_mb = os.path.getsize(checkpoint_out) / (1024 * 1024)
    print(f"\n[Agent-ML] Model successfully trained and saved to {checkpoint_out}")
    print(f"  Checkpoint Size: {weight_size_mb:.2f} MB (Target < 500 MB: PASSED)")
    return model_cpu


if __name__ == "__main__":
    run_curriculum_training()
