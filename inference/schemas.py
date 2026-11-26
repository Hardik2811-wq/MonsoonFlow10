"""Pydantic schemas and array decoding for MonsoonFlow-10 serving."""
from __future__ import annotations
import base64
from typing import Any, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field

# Constants matching competition contract
NUM_CHANNELS = 6
NUM_LATS = 32
NUM_LONS = 36
INPUT_DAYS = 3
DEFAULT_STEPS = 10


def decode_array(x: Any, expect_shape: Optional[tuple] = None) -> np.ndarray:
    """Accepts nested lists or {'b64': str, 'shape': list[int]}."""
    if isinstance(x, dict):
        if "b64" not in x or "shape" not in x:
            raise ValueError("array dict requires 'b64' and 'shape'")
        shape = tuple(int(s) for s in x["shape"])
        if any(s < 0 or s > 100000 for s in shape) or int(np.prod(shape)) > 20_000_000:
            raise ValueError("array dimensions exceed safety threshold")
        raw = base64.b64decode(x["b64"], validate=True)
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError("base64 byte buffer length does not match specified shape")
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape).astype("f4")
    else:
        arr = np.asarray(x, dtype="f4")

    if expect_shape is not None and arr.shape != tuple(expect_shape):
        raise ValueError(f"Shape {arr.shape} does not match expected {tuple(expect_shape)}")
    return arr


class HealthResponse(BaseModel):
    status: str = "ready"
    model_name: str = "MonsoonFlow-10"


class SimulateRequest(BaseModel):
    init_states: Any = Field(..., description="Nested lists or b64 dict for (3, 6, 32, 36) physical initial conditions")
    doy_sincos: List[float] = Field(..., description="[sin, cos] representing day of year cyclic encoding")
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=30, description="Number of autoregressive forecast steps")
    perturbation: Optional[Any] = Field(default=None, description="Optional physical shock delta of shape (6, 32, 36)")


class SimulateResponse(BaseModel):
    rollout: Any = Field(..., description="Forecast tensor of shape (steps, 6, 32, 36) in physical units")
    inference_ms: float = Field(..., description="Execution latency in milliseconds")
