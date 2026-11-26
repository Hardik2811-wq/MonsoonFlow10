"""Model and statistics artifact loader for offline serving."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


class ArtifactRegistry:
    """Singleton container preloading weights, statistics, and static terrain at boot."""

    def __init__(self):
        self.model = None
        self.stats = None
        self.static_terrain = None
        self.model_name = "MonsoonFlow-10"
        self._is_ready = False

    def load_artifacts(self) -> None:
        """Executed during server lifespan startup."""
        stats_path = CHECKPOINTS_DIR / "stats.json"
        if stats_path.is_file():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = None

        model_path = CHECKPOINTS_DIR / "model.pt"
        if model_path.is_file():
            try:
                import torch
                torch.set_num_threads(4)
                self.model = torch.load(model_path, map_location="cpu")
                if hasattr(self.model, "eval"):
                    self.model.eval()
                self.model_name = "MonsoonFlow-10-Torch"
            except Exception:
                self.model = None

        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready


registry = ArtifactRegistry()
