/**
 * api.ts - Wire interface schemas for root serve.py backend
 * Exactly matches root serve.py (FastAPI) and inference/schemas.py
 */

export interface HealthResponse {
  status: string;
  model_name?: string;
  device?: string;
  threads?: number;
  uptime_seconds?: number;
}

export interface SimulateRequest {
  init_states: number[][][][];      // (3, 6, 32, 36) in physical units
  doy_sincos: [number, number];      // [sin, cos] cyclic day-of-year representation
  steps: number;                    // 10
  perturbation?: number[][][] | null; // (6, 32, 36) physical perturbation delta
}

export interface SimulateResponse {
  rollout: number[][][][];          // (steps, 6, 32, 36) in physical units
  inference_ms: number;             // Execution latency in milliseconds
}

export interface ApiStatus {
  /** Health endpoint reachable right now. */
  serverOnline: boolean;
  /** Both the displayed baseline AND shock rollouts came from the live server. */
  isLive: boolean;
  /** Reason the synthetic fallback was used for the displayed rollouts, if any. */
  fallbackReason: string | null;
  /** Day-0 initial condition description and whether it is synthetic. */
  initialStateLabel: string;
  initialStateIsSynthetic: boolean;
  isSimulating: boolean;
  latencyMs: number;
  endpoint: string;
  modelName: string;
  lastSimulatedAt: Date | null;
  errorMessage: string | null;
}
