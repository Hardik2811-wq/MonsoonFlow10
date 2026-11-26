/**
 * api.ts - HTTP client for root serve.py
 * - POST /simulate with init_states: (3, 6, 32, 36), doy_sincos: [sin, cos], steps: 10, perturbation: (6, 32, 36) | null
 * - GET /health returning {"status": "ready", "model_name": ...}
 *
 * If the server cannot be reached or errors, a local SYNTHETIC diffusion rollout is returned and
 * flagged with isLiveBackend=false plus the reason, so the UI never presents it as model output.
 */

import type { ChannelTensor, SimulationRollout } from '../types/climate';
import type { HealthResponse, SimulateRequest, SimulateResponse } from '../types/api';
import { simulateOfflineRollout } from './syntheticData';

// 127.0.0.1, not localhost: on Windows 'localhost' tries IPv6 ::1 first and stalls before falling back
const BASE_URL = 'http://127.0.0.1:8000';

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

export interface SimulationResult {
  rollout: SimulationRollout;
  inferenceTimeMs: number;
  isLiveBackend: boolean;
  /** Why the synthetic fallback was used (null when the rollout came from the live server). */
  fallbackReason: string | null;
}

export async function runSimulation(
  history3Day: ChannelTensor[],
  doySincos: [number, number],
  perturbation: ChannelTensor | null
): Promise<SimulationResult> {
  const startTime = performance.now();

  const payload: SimulateRequest = {
    init_states: history3Day,
    doy_sincos: doySincos,
    steps: 10,
    perturbation,
  };

  let fallbackReason: string;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5.0 s competition latency ceiling

    const res = await fetch(`${BASE_URL}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data: SimulateResponse = await res.json();
      const elapsed = data.inference_ms ?? performance.now() - startTime;
      return {
        rollout: data.rollout,
        inferenceTimeMs: Math.round(elapsed * 10) / 10,
        isLiveBackend: true,
        fallbackReason: null,
      };
    }
    fallbackReason = `server returned HTTP ${res.status}`;
  } catch (err: unknown) {
    fallbackReason =
      err instanceof DOMException && err.name === 'AbortError'
        ? 'server timed out (> 5 s)'
        : 'server unreachable';
  }

  const fallbackRollout = simulateOfflineRollout(history3Day[2], perturbation);
  return {
    rollout: fallbackRollout,
    inferenceTimeMs: Math.round(performance.now() - startTime),
    isLiveBackend: false,
    fallbackReason,
  };
}
