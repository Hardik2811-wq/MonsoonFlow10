/**
 * api.ts - Wire interface schemas for root serve.py backend
 */

export interface HealthResponse {
  status: string;
  uptime_seconds?: number;
  device?: string;
  threads?: number;
  model_loaded?: boolean;
}

export interface SimulateRequest {
  initial_condition: number[][][]; // [6, 32, 36] physical units
  shock?: number[][][] | null;     // [6, 32, 36] physical delta perturbations
}

export interface SimulateResponse {
  prediction: number[][][][];      // [10, 6, 32, 36] physical units
  inference_time_ms: number;
  offline_cpu?: boolean;
  status?: string;
}

export interface ApiStatus {
  isLive: boolean;
  isSimulating: boolean;
  latencyMs: number;
  endpoint: string;
  lastSimulatedAt: Date | null;
  errorMessage: string | null;
}
