/**
 * physicsAudit.ts - Lean <2ms atmospheric physics & vortex diagnostic on 32x36 tensors.
 * Grounded in Sha et al. (2025) / Chapman et al. (2026) and Sahu et al. (2025):
 *   - Tracks Peak 10-m Relative Vorticity (zeta10 = dv10/dx - du10/dy) & Min Central MSL (hPa).
 *   - Checks open-boundary moisture plausibility (tp >= 0, tcwv > 0, finite) and 2D moisture
 *     convergence tendency residual R_q (noting open-boundary advection, not closed-box rescaling).
 */

import { GRID_CONFIG, LATS, type ChannelTensor } from '../types/climate';

export interface PhysicsAuditReport {
  /** Peak |zeta10| in 10^-5 s^-1 (cyclonic/anticyclonic 10-m relative vorticity) */
  peakVorticity1e5: number;
  /** Signed zeta10 at the peak location in 10^-5 s^-1 */
  signedPeakVorticity1e5: number;
  /** Domain minimum mean sea-level pressure in hPa */
  minMslHpa: number;
  /** Open-boundary 2D moisture-convergence tendency residual R_q (mm/day) */
  moistureResidualMmDay: number;
  /** True if tp >= 0, tcwv > 0, all finite, and R_q within open-boundary plausibility */
  plausible: boolean;
  status: 'PASS' | 'CAUTION';
  note: string;
}

const NLAT = GRID_CONFIG.nLat;
const NLON = GRID_CONFIG.nLon;
const DY_M = 111_000 * GRID_CONFIG.resolutionDeg;
const SEC_PER_DAY = 86_400;

export function computePhysicsAudit(
  frame: ChannelTensor,
  prevFrame?: ChannelTensor
): PhysicsAuditReport {
  const tp = frame[1];
  const u10 = frame[2];
  const v10 = frame[3];
  const msl = frame[4];
  const tcwv = frame[5];
  const prevTcwv = prevFrame ? prevFrame[5] : null;

  let minMslPa = Infinity;
  let allFinite = true;
  let nonNegTp = true;
  let posTcwv = true;

  let sumTp = 0;
  let sumDqDt = 0;
  let countAll = 0;

  for (let i = 0; i < NLAT; i++) {
    for (let j = 0; j < NLON; j++) {
      const p = tp[i][j];
      const u = u10[i][j];
      const v = v10[i][j];
      const m = msl[i][j];
      const q = tcwv[i][j];

      if (
        !Number.isFinite(p) ||
        !Number.isFinite(u) ||
        !Number.isFinite(v) ||
        !Number.isFinite(m) ||
        !Number.isFinite(q)
      ) {
        allFinite = false;
      }
      if (p < -1e-6) nonNegTp = false;
      if (q <= 0) posTcwv = false;
      if (m < minMslPa) minMslPa = m;

      sumTp += p;
      if (prevTcwv) sumDqDt += q - prevTcwv[i][j];
      countAll++;
    }
  }

  let peakAbsZeta = 0;
  let signedPeakZeta = 0;
  let sumDivQ = 0;
  let countInt = 0;

  for (let i = 1; i < NLAT - 1; i++) {
    const dx = DY_M * Math.max(0.2, Math.cos((LATS[i] * Math.PI) / 180));
    const twoDx = 2 * dx;
    const twoDy = 2 * DY_M;
    for (let j = 1; j < NLON - 1; j++) {
      const dvdx = (v10[i][j + 1] - v10[i][j - 1]) / twoDx;
      const dudy = (u10[i + 1][j] - u10[i - 1][j]) / twoDy;
      const zeta = dvdx - dudy;
      if (Math.abs(zeta) > peakAbsZeta) {
        peakAbsZeta = Math.abs(zeta);
        signedPeakZeta = zeta;
      }

      const dqudx = (tcwv[i][j + 1] * u10[i][j + 1] - tcwv[i][j - 1] * u10[i][j - 1]) / twoDx;
      const dqvdy = (tcwv[i + 1][j] * v10[i + 1][j] - tcwv[i - 1][j] * v10[i - 1][j]) / twoDy;
      sumDivQ += dqudx + dqvdy;
      countInt++;
    }
  }

  const meanTp = countAll > 0 ? sumTp / countAll : 0;
  const meanDqDt = countAll > 0 ? sumDqDt / countAll : 0;
  // Convert surface-wind moisture flux divergence (kg m^-2 s^-1) to mm/day scale
  const meanDivQMmDay = countInt > 0 ? (sumDivQ / countInt) * SEC_PER_DAY * 0.25 : 0;
  const residualRq = Math.abs(meanDqDt + meanDivQMmDay + meanTp);

  const minMslHpa = Number.isFinite(minMslPa) ? minMslPa / 100 : 1013.2;
  const plausible = allFinite && nonNegTp && posTcwv && minMslHpa >= 940 && residualRq < 45;

  return {
    peakVorticity1e5: Math.round(peakAbsZeta * 1e5 * 10) / 10,
    signedPeakVorticity1e5: Math.round(signedPeakZeta * 1e5 * 10) / 10,
    minMslHpa: Math.round(minMslHpa * 10) / 10,
    moistureResidualMmDay: Math.round(residualRq * 10) / 10,
    plausible,
    status: plausible ? 'PASS' : 'CAUTION',
    note: plausible
      ? 'Open-boundary advection active (tp≥0, tcwv>0)'
      : 'Check boundary flux / non-physical extremum',
  };
}
