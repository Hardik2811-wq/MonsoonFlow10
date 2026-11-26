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
  isLive: boolean;
  isSimulating: boolean;
  latencyMs: number;
  endpoint: string;
  modelName: string;
  lastSimulatedAt: Date | null;
  errorMessage: string | null;
}
