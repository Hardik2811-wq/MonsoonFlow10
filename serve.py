"""Root inference server for MonsoonFlow-10.
Listens on 0.0.0.0:8000 (configurable via --port).
Provides /health and /simulate endpoints under strict offline constraints.
"""
from __future__ import annotations
import argparse
from contextlib import asynccontextmanager
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.loader import registry
from inference.predictor import simulate, get_model_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model weights, statistics, and static terrain at startup."""
    registry.load_artifacts()
    yield


app = FastAPI(
    title="MonsoonFlow-10 Inference Service",
    description="Offline spatiotemporal climate digital twin inference server",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local frontend communication (e.g. Vite on localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
@app.post("/health", status_code=status.HTTP_200_OK)
def health_endpoint():
    """Health check endpoint required by competition grader and frontend."""
    return {"status": "ready", "model_name": get_model_name()}


@app.post("/simulate", status_code=status.HTTP_200_OK)
async def simulate_endpoint(request: Request):
    """Execute 10-step spatiotemporal weather rollout.

    Accepts physical units:
      - init_states: (3, 6, 32, 36)
      - doy_sincos: [sin, cos]
      - steps: integer (default 10)
      - perturbation: optional (6, 32, 36) delta shock
    Returns:
      - rollout: (steps, 6, 32, 36) in physical units
      - inference_ms: latency in milliseconds
    """
    body = await request.json()
    init_states = body.get("init_states")
    doy_sincos = body.get("doy_sincos", [0.0, 1.0])
    steps = int(body.get("steps", 10))
    perturbation = body.get("perturbation")

    out, inference_ms = simulate(
        init_states_raw=init_states,
        doy_sincos=doy_sincos,
        steps=steps,
        perturbation_raw=perturbation,
    )

    return {
        "rollout": np.asarray(out, dtype="f4").tolist(),
        "inference_ms": float(inference_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="MonsoonFlow-10 Root Serving Entrypoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
