"""Model, statistics, and static terrain preloading registry for offline serving."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple
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
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
                mean_list = self.stats["mean"]
                std_list = self.stats["std"]
                self.mean_np = np.array(mean_list, dtype=np.float32).reshape(1, 6, 1, 1)
                self.std_np = np.array(std_list, dtype=np.float32).reshape(1, 6, 1, 1)
                try:
                    import torch
                    self.mean_th = torch.from_numpy(self.mean_np)
                    self.std_th = torch.from_numpy(self.std_np)
                except Exception:
                    pass
            except Exception:
                self.stats = None

        # 2. Load Static Environmental Fields (land_sea_mask, orography)
        static_path = CHECKPOINTS_DIR / "static.npy"
        if static_path.is_file():
            try:
                raw_static = np.load(static_path).astype(np.float32)  # (2, 32, 36)
                lsm = raw_static[0:1]
                oro_norm = raw_static[1:2] / 8000.0
                self.static_terrain_np = np.concatenate([lsm, oro_norm], axis=0)  # (2, 32, 36)
                try:
                    import torch
                    self.static_terrain_th = torch.from_numpy(self.static_terrain_np).unsqueeze(0)  # (1, 2, 32, 36)
                except Exception:
                    pass
            except Exception:
                self.static_terrain_np = None

        # 3. Load Trained Model Checkpoint
        model_path = CHECKPOINTS_DIR / "model.pt"
        if model_path.is_file():
            try:
                import torch
                torch.set_num_threads(4)
                loaded = torch.load(model_path, map_location="cpu", weights_only=False)
                if hasattr(loaded, "eval"):
                    loaded.eval()
                    self.model = loaded
                elif isinstance(loaded, dict) and "state_dict" in loaded:
                    # state_dict format - held ready for architecture binding
                    self.model = loaded
                else:
                    self.model = loaded
                self.model_name = "MonsoonFlow-10-Neural"
            except Exception:
                self.model = None

        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready


registry = ArtifactRegistry()
