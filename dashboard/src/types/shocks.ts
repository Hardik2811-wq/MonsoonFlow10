/**
 * shocks.ts - Shock configurations and Gaussian smoothing specifications
 */

import type { ChannelId } from './climate';

export interface GaussianShockSpec {
  centerLat: number;
  centerLon: number;
  radiusKm: number;
  sigmaKm: number; // default ~300 km (~2.5 grid cells)
  channel: ChannelId;
  magnitude: number; // in physical wire units (e.g. +3.0 K, -5 m/s)
}

export type PresetShockId = 'WARM_BAY_OF_BENGAL' | 'MONSOON_STALL' | 'WESTERN_GHATS_DELUGE' | 'ZERO_SHOCK';

export interface ShockPreset {
  id: PresetShockId;
  name: string;
  category: 'Thermal' | 'Dynamical' | 'Hydrological' | 'Baseline';
  description: string;
  spec: GaussianShockSpec | null;
  boundsWarning?: string;
}

export const SHOCK_PRESETS: ShockPreset[] = [
  {
    id: 'WARM_BAY_OF_BENGAL',
    name: 'Warm Bay of Bengal',
    category: 'Thermal',
    description: '+3.0 K SST/T2M warming anomaly centered over Bay of Bengal (14.5°N, 88.5°E), intensifying moisture convection',
    spec: {
      centerLat: 14.5,
      centerLon: 88.5,
      radiusKm: 500,
      sigmaKm: 300,
      channel: 't2m',
      magnitude: 3.0,
    },
  },
  {
    id: 'MONSOON_STALL',
    name: 'Arabian Sea Monsoon Stall',
    category: 'Dynamical',
    description: '-40% zonal wind deceleration (-6.0 m/s u10) over eastern Arabian Sea (14.0°N, 70.0°E), suppressing moisture transport',
    spec: {
      centerLat: 14.0,
      centerLon: 70.0,
      radiusKm: 450,
      sigmaKm: 280,
      channel: 'u10',
      magnitude: -6.0,
    },
  },
  {
    id: 'WESTERN_GHATS_DELUGE',
    name: 'Western Ghats Deluge',
    category: 'Hydrological',
    description: '+30 mm/day extreme precipitation pulse over Konkan/Goa coast (15.5°N, 73.5°E)',
    spec: {
      centerLat: 15.5,
      centerLon: 73.5,
      radiusKm: 350,
      sigmaKm: 220,
      channel: 'tp',
      magnitude: 30.0,
    },
  },
  {
    id: 'ZERO_SHOCK',
    name: 'Zero Shock / Reset',
    category: 'Baseline',
    description: 'No counterfactual perturbation (P = 0). Produces exact baseline rollout within 10⁻⁶ numerical tolerance',
    spec: null,
  },
];
