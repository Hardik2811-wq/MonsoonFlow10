/**
 * api.ts - Resilient HTTP client for root serve.py
 * Fully compliant with root serve.py wire contracts:
 * - POST /simulate with init_states: (3, 6, 32, 36), doy_sincos: [sin, cos], steps: 10
 * - GET /health returning {"status": "ready", "model_name": "MonsoonFlow-10"}
 */

import type { ChannelTensor, SimulationRollout } from '../types/climate';
import type { HealthResponse, SimulateRequest, SimulateResponse } from '../types/api';
import type { GaussianShockSpec } from '../types/shocks';
import { generateGaussianShock2D } from './gaussianBrush';
import {
  createZeroTensor,
  generate3DayClimatologicalHistory,
  simulateOfflineRollout,
} from './syntheticData';

const BASE_URL = 'http://localhost:8000';

export async function checkBackendHealth(): Promise<{ isOnline: boolean; info?: HealthResponse }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);

    const res = await fetch(`${BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data: HealthResponse = await res.json();
      return { isOnline: true, info: data };
    }
    return { isOnline: false };
  } catch {
    return { isOnline: false };
  }
}

export async function runSimulation(
  history3Day: ChannelTensor[],
  shockSpec: GaussianShockSpec | null
): Promise<{
  rollout: SimulationRollout;
  inferenceTimeMs: number;
  isLiveBackend: boolean;
}> {
  const startTime = performance.now();

  // Prepare perturbation tensor [6, 32, 36]
  let shockTensor: ChannelTensor | null = null;
  if (shockSpec && shockSpec.magnitude !== 0) {
    shockTensor = createZeroTensor();
    const shockGrid = generateGaussianShock2D(shockSpec);
    const chIdx = ['t2m', 'tp', 'u10', 'v10', 'msl', 'tcwv'].indexOf(shockSpec.channel);
    if (chIdx >= 0) {
      shockTensor[chIdx] = shockGrid;
    }
  }

  // Peak monsoon day of year sin/cos (~July 15, DOY 196: sin(2pi*196/365) = -0.197, cos = -0.980)
  const payload: SimulateRequest = {
    init_states: history3Day,
    doy_sincos: [-0.197, -0.980],
    steps: 10,
    perturbation: shockTensor,
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // Strict 5.0s offline CPU limit

    const res = await fetch(`${BASE_URL}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data: SimulateResponse = await res.json();
      const elapsed = data.inference_ms || Math.round(performance.now() - startTime);
      return {
        rollout: data.rollout,
        inferenceTimeMs: Math.round(elapsed * 10) / 10,
        isLiveBackend: true,
      };
    }
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  } catch {
    // Clean offline fallback using local synoptic engine
    const elapsed = Math.round(performance.now() - startTime);
    const fallbackRollout = simulateOfflineRollout(history3Day[2], shockSpec);
    return {
      rollout: fallbackRollout,
      inferenceTimeMs: elapsed,
      isLiveBackend: false,
    };
  }
}

export { generate3DayClimatologicalHistory };
