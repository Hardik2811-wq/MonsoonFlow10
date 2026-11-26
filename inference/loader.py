"""Model, statistics, and static terrain preloading registry for offline serving."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


class ArtifactRegistry:
    """Singleton registry preloading weights, normalization stats, and terrain."""

    def __init__(self):
        self.model = None
        self.stats = None
        self.mean_np: Optional[np.ndarray] = None
        self.std_np: Optional[np.ndarray] = None
        self.mean_th = None
        self.std_th = None
        self.static_terrain_np: Optional[np.ndarray] = None
        self.static_terrain_th = None
        self.model_name = "MonsoonFlow-10"
        self._is_ready = False

    def load_artifacts(self) -> None:
        """Executed during server startup lifespan."""
        # 1. Load Normalization Statistics
        stats_path = CHECKPOINTS_DIR / "stats.json"
        if stats_path.is_file():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.stats = json.load(f)
            self.mean_np = np.array(self.stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
            self.std_np = np.array(self.stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
            import torch
            self.mean_th = torch.from_numpy(self.mean_np)
            self.std_th = torch.from_numpy(self.std_np)

        # 2. Load Static Environmental Fields (land_sea_mask, orography)
        static_path = CHECKPOINTS_DIR / "static.npy"
        if static_path.is_file():
            raw_static = np.load(static_path).astype(np.float32)  # (2, 32, 36)
            lsm = raw_static[0:1]
            oro_norm = raw_static[1:2] / 8000.0
            self.static_terrain_np = np.concatenate([lsm, oro_norm], axis=0)  # (2, 32, 36)
            import torch
            self.static_terrain_th = torch.from_numpy(self.static_terrain_np).unsqueeze(0)  # (1, 2, 32, 36)

        # 3. Load Trained MF10 Model Checkpoint
        model_path = CHECKPOINTS_DIR / "model.pt"
        if model_path.is_file():
            import torch
            from models.mf10 import MonsoonFlow10

            torch.set_num_threads(4)
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                state_dict = loaded.get("state_dict", loaded)
                try:
                    model = MonsoonFlow10(
                        in_channels=10,
                        dynamic_channels=6,
                        hidden_dim=64,
                        num_lstm_layers=2,
                        stats=self.stats,
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.model = model
                    self.model_name = "MonsoonFlow-10"
                except RuntimeError as e:
                    print(f"[loader] Warning: checkpoint mismatch ({type(e).__name__}); awaiting retrained weights.")
                    self.model = None
                    self.model_name = "MonsoonFlow-10-PendingWeights"
            elif hasattr(loaded, "eval"):
                loaded.eval()
                self.model = loaded
                self.model_name = "MonsoonFlow-10"

            # Warmup forward pass to prime CPU caches
            if self.model is not None:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 10, 32, 36, dtype=torch.float32)
                    _ = self.model(dummy, steps=10)

        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready


registry = ArtifactRegistry()
