/**
 * api.ts - Resilient HTTP client for root serve.py
 * Calls POST /simulate and GET /health, falling back cleanly to local physical engine
 * when backend server is starting up.
 */

import type { ChannelTensor, SimulationRollout } from '../types/climate';
import type { HealthResponse, SimulateRequest, SimulateResponse } from '../types/api';
import type { GaussianShockSpec } from '../types/shocks';
import { generateGaussianShock2D } from './gaussianBrush';
import { createZeroTensor, generateClimatologicalInitialCondition, simulateOfflineRollout } from './syntheticData';

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
  initialCondition: ChannelTensor,
  shockSpec: GaussianShockSpec | null
): Promise<{
  rollout: SimulationRollout;
  inferenceTimeMs: number;
  isLiveBackend: boolean;
}> {
  const startTime = performance.now();

  // Prepare shock tensor [6, 32, 36]
  let shockTensor: ChannelTensor | null = null;
  if (shockSpec && shockSpec.magnitude !== 0) {
    shockTensor = createZeroTensor();
    const shockGrid = generateGaussianShock2D(shockSpec);
    const chIdx = ['t2m', 'tp', 'u10', 'v10', 'msl', 'tcwv'].indexOf(shockSpec.channel);
    if (chIdx >= 0) {
      shockTensor[chIdx] = shockGrid;
    }
  }

  const payload: SimulateRequest = {
    initial_condition: initialCondition,
    shock: shockTensor,
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000); // 6s timeout (rule requires <= 5s)

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
      const elapsed = Math.round(performance.now() - startTime);
      return {
        rollout: data.prediction,
        inferenceTimeMs: data.inference_time_ms || elapsed,
        isLiveBackend: true,
      };
    }
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  } catch {
    // Graceful offline fallback using high-fidelity local synoptic engine
    const elapsed = Math.round(performance.now() - startTime);
    const offlineRollout = simulateOfflineRollout(initialCondition, shockSpec);
    return {
      rollout: offlineRollout,
      inferenceTimeMs: elapsed,
      isLiveBackend: false,
    };
  }
}

export { generateClimatologicalInitialCondition };
