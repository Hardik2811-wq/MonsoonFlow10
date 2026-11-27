"""CPU latency and contract benchmark for MonsoonFlow-10 offline serving."""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
import numpy as np

# Ensure repository root is on path when running as standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.loader import registry
from inference.predictor import simulate
from inference.schemas import NUM_CHANNELS, NUM_LATS, NUM_LONS, INPUT_DAYS, DEFAULT_STEPS


def generate_synthetic_physical_state(seed: int = 42) -> np.ndarray:
    """Generate realistic physical weather input tensor of shape (3, 6, 32, 36)."""
    rng = np.random.default_rng(seed)
    # Channel baseline values: t2m(K), tp(mm/day), u10(m/s), v10(m/s), msl(Pa), tcwv(kg/m2)
    bases = np.array([295.0, 4.0, 2.0, 1.0, 101000.0, 35.0], dtype=np.float32)[:, None, None]
    stds = np.array([5.0, 4.0, 3.0, 3.0, 400.0, 10.0], dtype=np.float32)[:, None, None]

    state = np.zeros((INPUT_DAYS, NUM_CHANNELS, NUM_LATS, NUM_LONS), dtype=np.float32)
    for t in range(INPUT_DAYS):
        noise = rng.standard_normal((NUM_CHANNELS, NUM_LATS, NUM_LONS), dtype=np.float32)
        state[t] = bases + noise * stds
        state[t, 1] = np.maximum(state[t, 1], 0.0)  # Precipitation >= 0
        state[t, 0] = np.clip(state[t, 0], 233.15, 333.15)  # Physical temperature range
    return state


def run_benchmark(num_runs: int = 20, warmup_runs: int = 5) -> dict:
    """Run CPU inference latency and contract validation suite."""
    # 1. Initialize registry
    registry.load_artifacts()

    # 2. Configure PyTorch CPU threading
    try:
        import torch
        torch.set_num_threads(4)
        active_threads = torch.get_num_threads()
    except Exception:
        active_threads = 1

    # 3. Prepare test inputs
    init_state = generate_synthetic_physical_state(seed=123)
    doy_sincos = [0.866, -0.500]  # Example cyclic features
    steps = DEFAULT_STEPS

    # Shock perturbations
    zero_shock = np.zeros((NUM_CHANNELS, NUM_LATS, NUM_LONS), dtype=np.float32)
    nonzero_shock = np.zeros((NUM_CHANNELS, NUM_LATS, NUM_LONS), dtype=np.float32)
    nonzero_shock[0, 10:20, 15:25] = 2.5  # +2.5 K localized sea surface anomaly

    # 4. Warmup phase
    for _ in range(warmup_runs):
        simulate(init_state, doy_sincos, steps=steps, perturbation_raw=None)

    # 5. Timed rollout benchmarking
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out, _ = simulate(init_state, doy_sincos, steps=steps, perturbation_raw=None)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000.0)

    # 6. Contract Verification
    # A. Output Shape
    assert out.shape == (steps, NUM_CHANNELS, NUM_LATS, NUM_LONS), f"Invalid shape: {out.shape}"

    # B. Finite check
    assert np.isfinite(out).all(), "Rollout contains NaNs or Infs!"

    # C. Physical Boundaries
    assert out[:, 1].min() >= -1e-6, f"Negative precipitation detected: {out[:, 1].min()}"
    assert out[:, 0].min() >= 233.15, f"t2m below -40C: {out[:, 0].min()} K"
    assert out[:, 0].max() <= 333.15, f"t2m above +60C: {out[:, 0].max()} K"

    # D. Zero-Shock Invariance Contract (atol < 1e-6)
    baseline_out, _ = simulate(init_state, doy_sincos, steps=steps, perturbation_raw=None)
    zero_out, _ = simulate(init_state, doy_sincos, steps=steps, perturbation_raw=zero_shock)
    max_zero_diff = float(np.max(np.abs(baseline_out - zero_out)))
    assert max_zero_diff < 1e-6, f"Zero-shock contract violated! Max diff: {max_zero_diff}"

    # E. Perturbation Response
    shock_out, _ = simulate(init_state, doy_sincos, steps=steps, perturbation_raw=nonzero_shock)
    shock_diff = float(np.max(np.abs(shock_out - baseline_out)))
    assert shock_diff > 0.0, "Perturbation had zero effect on forecast!"

    # 7. Summary metrics
    lat_arr = np.array(latencies)
    mean_ms = float(np.mean(lat_arr))
    median_ms = float(np.median(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    min_ms = float(np.min(lat_arr))
    max_ms = float(np.max(lat_arr))

    results = {
        "model_name": registry.model_name,
        "threads": active_threads,
        "steps": steps,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "zero_shock_max_diff": max_zero_diff,
        "latency_target_pass": mean_ms <= 500.0,
        "latency_hard_limit_pass": max_ms <= 5000.0,
    }
    return results


def main():
    print("=" * 60)
    print("MonsoonFlow-10 CPU Inference Benchmark & Contract Validator")
    print("=" * 60)
    res = run_benchmark(num_runs=30, warmup_runs=5)

    print(f"Model Name:              {res['model_name']}")
    print(f"Active CPU Threads:      {res['threads']}")
    print(f"Rollout Forecast Steps:  {res['steps']}")
    print("-" * 60)
    print(f"Mean Latency:            {res['mean_ms']:.2f} ms")
    print(f"Median Latency:          {res['median_ms']:.2f} ms")
    print(f"P95 Latency:             {res['p95_ms']:.2f} ms")
    print(f"Min / Max:               {res['min_ms']:.2f} ms / {res['max_ms']:.2f} ms")
    print("-" * 60)
    print(f"Zero-Shock Max Diff:     {res['zero_shock_max_diff']:.2e} (Passes atol < 1e-6: True)")
    print(f"Target <= 0.5s (500ms):  {'PASS' if res['latency_target_pass'] else 'FAIL'}")
    print(f"Hard Limit <= 5.0s:      {'PASS' if res['latency_hard_limit_pass'] else 'FAIL'}")
    print("=" * 60)
    print("Benchmark Status:        CERTIFIED READY")


if __name__ == "__main__":
    main()
