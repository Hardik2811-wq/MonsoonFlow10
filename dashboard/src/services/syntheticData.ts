/**
 * syntheticData.ts - High-fidelity synthetic monsoon climate simulation engine
 * Provides realistic initial conditions and physics-informed 10-day rollouts
 * for immediate offline exploration and automatic fallback when backend is starting up.
 */

import { GRID_CONFIG, type ChannelTensor, type SimulationRollout } from '../types/climate';
import type { GaussianShockSpec } from '../types/shocks';
import { generateGaussianShock2D } from './gaussianBrush';

export function createZeroTensor(): ChannelTensor {
  return Array.from({ length: 6 }, () =>
    Array.from({ length: GRID_CONFIG.nLat }, () =>
      Array.from({ length: GRID_CONFIG.nLon }, () => 0)
    )
  );
}

/**
 * Generates an authentic Day 0 monsoon synoptic state:
 * - Arabian Sea low-level jet (strong south-westerly winds)
 * - Monsoon trough over Indo-Gangetic plain (lower MSL)
 * - Heavy orographic rainfall along Western Ghats and Meghalaya
 * - High tropical temperatures in northwest (Rajasthan), cooler in southern ocean
 */
export function generateClimatologicalInitialCondition(): ChannelTensor {
  const t2m: number[][] = [];
  const tp: number[][] = [];
  const u10: number[][] = [];
  const v10: number[][] = [];
  const msl: number[][] = [];
  const tcwv: number[][] = [];

  for (let i = 0; i < GRID_CONFIG.nLat; i++) {
    const lat = GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg;
    const rowT2m: number[] = [];
    const rowTp: number[] = [];
    const rowU10: number[] = [];
    const rowV10: number[] = [];
    const rowMsl: number[] = [];
    const rowTcwv: number[] = [];

    for (let j = 0; j < GRID_CONFIG.nLon; j++) {
      const lon = GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg;

      // 1. Temperature: Warmer in NW India (lat ~26-30, lon ~70-75)
      const distNW = Math.hypot(lat - 28.0, lon - 72.0);
      const tempK = 299.15 + 8.0 * Math.exp(-(distNW * distNW) / 100) - (lat - 20) * 0.25;
      rowT2m.push(Math.round(tempK * 10) / 10);

      // 2. Precipitation: Western Ghats (lat 9-18, lon 73-76) and Northeast (lat 24-28, lon 90-95)
      const distWG = Math.hypot(lat - 14.5, lon - 74.5);
      const distNE = Math.hypot(lat - 25.5, lon - 92.5);
      const precip =
        35.0 * Math.exp(-(distWG * distWG) / 12) +
        28.0 * Math.exp(-(distNE * distNE) / 18) +
        Math.max(0, 3.0 * Math.sin((lon - 70) * 0.15));
      rowTp.push(Math.max(0, Math.round(precip * 10) / 10));

      // 3. Low-Level Jet (Findlater Jet): Strong SW winds in Arabian Sea (lat 10-18, lon 65-75)
      const distLLJ = Math.hypot(lat - 14.0, lon - 68.0);
      const u = 6.0 + 10.0 * Math.exp(-(distLLJ * distLLJ) / 45);
      const v = 4.0 + 7.0 * Math.exp(-(distLLJ * distLLJ) / 45);
      rowU10.push(Math.round(u * 10) / 10);
      rowV10.push(Math.round(v * 10) / 10);

      // 4. Monsoon Trough (MSL): Low pressure over central/northern India
      const distTrough = Math.hypot(lat - 24.0, lon - 80.0);
      const pressure = 100800 - 800 * Math.exp(-(distTrough * distTrough) / 80);
      rowMsl.push(Math.round(pressure));

      // 5. TCWV: High moisture over Bay of Bengal and Arabian Sea (55-65 kg/m²)
      const moist = 52.0 + 12.0 * Math.sin(lat * 0.08) - (lat > 28 ? 15 : 0);
      rowTcwv.push(Math.round(Math.max(15, moist) * 10) / 10);
    }

    t2m.push(rowT2m);
    tp.push(rowTp);
    u10.push(rowU10);
    v10.push(rowV10);
    msl.push(rowMsl);
    tcwv.push(rowTcwv);
  }

  return [t2m, tp, u10, v10, msl, tcwv];
}

/**
 * Autoregressively rolls out 10 steps of atmospheric state.
 * Models physical dispersion, eastward synoptic propagation, and advection.
 */
export function simulateOfflineRollout(
  initialState: ChannelTensor,
  shockSpec: GaussianShockSpec | null
): SimulationRollout {
  const rollout: SimulationRollout = [];

  // Deep clone initial state
  let currentState: ChannelTensor = JSON.parse(JSON.stringify(initialState));

  // Compute shock delta if active
  let shockGrid: number[][] | null = null;
  if (shockSpec && shockSpec.magnitude !== 0) {
    shockGrid = generateGaussianShock2D(shockSpec);
    // Apply shock at Day 1 (immediate perturbation to corresponding channel)
    const chIdx = ['t2m', 'tp', 'u10', 'v10', 'msl', 'tcwv'].indexOf(shockSpec.channel);
    if (chIdx >= 0) {
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          currentState[chIdx][i][j] += shockGrid[i][j];
          if (shockSpec.channel === 'tp') {
            currentState[chIdx][i][j] = Math.max(0, currentState[chIdx][i][j]);
          }
        }
      }
    }
  }

  // Day 1 to Day 10 rollouts
  for (let day = 1; day <= 10; day++) {
    const nextState: ChannelTensor = JSON.parse(JSON.stringify(currentState));

    // Advection & dispersion simulation across grid
    for (let c = 0; c < 6; c++) {
      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          const val = currentState[c][i][j];
          // East-northeast synoptic propagation (~1.5 cells/day)
          const westVal = j > 0 ? currentState[c][i][j - 1] : val;
          const southVal = i > 0 ? currentState[c][i - 1][j] : val;
          const eastVal = j < GRID_CONFIG.nLon - 1 ? currentState[c][i][j + 1] : val;
          const northVal = i < GRID_CONFIG.nLat - 1 ? currentState[c][i + 1][j] : val;

          // Diffusion + advection laplacian
          const laplacian = 0.25 * (westVal + southVal + eastVal + northVal) - val;
          const advection = 0.08 * (westVal - val);

          let newVal = val + 0.3 * laplacian + advection;

          // Precipitation non-negativity constraint
          if (c === 1) {
            newVal = Math.max(0, newVal);
          }

          nextState[c][i][j] = Math.round(newVal * 100) / 100;
        }
      }
    }

    rollout.push(nextState);
    currentState = nextState;
  }

  return rollout;
}
