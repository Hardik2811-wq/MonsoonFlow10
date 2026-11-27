/**
 * impactCalculator.ts - Calculates Flood Risk Index and Crop Moisture Stress Index
 * across India's key meteorological and agricultural river basins.
 */

import { GRID_CONFIG, LATS, type ChannelTensor, type Grid2D } from '../types/climate';
import { MONSOON_BASINS, type BasinImpact, type DownstreamImpactSummary } from '../types/impacts';
import { latLonToGrid } from './physicalUnits';

export function calculateDownstreamImpacts(
  state: ChannelTensor,
  baselineState?: ChannelTensor,
  landSeaMask?: Grid2D
): DownstreamImpactSummary {
  const tp = state[1]; // Precipitation grid [32, 36] in mm/day
  const t2m = state[0]; // Temperature grid [32, 36] in K
  const baseTp = baselineState ? baselineState[1] : null;
  const isLand = (i: number, j: number) => (landSeaMask ? landSeaMask[i][j] >= 0.5 : true);

  // Grid-wide counts over actual India landmass (land_sea_mask >= 0.5 when available)
  let extremeCount = 0; // tp >= 50 mm/day
  let heavyCount = 0; // tp >= 10 mm/day (includes extreme cells)
  let drySpellCount = 0; // tp < 2 mm/day under t2m > 305 K
  let extremeAreaKm2 = 0;
  const kmPerDeg = GRID_CONFIG.approxKmPerDeg;
  for (let i = 0; i < GRID_CONFIG.nLat; i++) {
    // 1 deg x 1 deg cell area shrinks with cos(latitude)
    const cellAreaKm2 = kmPerDeg * kmPerDeg * Math.cos((LATS[i] * Math.PI) / 180) * GRID_CONFIG.resolutionDeg ** 2;
    for (let j = 0; j < GRID_CONFIG.nLon; j++) {
      if (!isLand(i, j)) continue;
      const p = tp[i][j];
      if (p >= 50) {
        extremeCount++;
        extremeAreaKm2 += cellAreaKm2;
      }
      if (p >= 10) heavyCount++;
      if (p < 2.0 && t2m[i][j] > 305) drySpellCount++;
    }
  }

  const basins: BasinImpact[] = MONSOON_BASINS.map((b) => {
    let sumPrecip = 0;
    let sumBasePrecip = 0;
    let cellCount = 0;
    let basinHeavyCount = 0;
    let basinExtremeCount = 0;
    let basinMaxP = 0;

    const gMin = latLonToGrid(b.latRange[0], b.lonRange[0]);
    const gMax = latLonToGrid(b.latRange[1], b.lonRange[1]);

    if (gMin && gMax) {
      for (let i = gMin.i; i <= gMax.i; i++) {
        for (let j = gMin.j; j <= gMax.j; j++) {
          if (!isLand(i, j)) continue;
          const p = tp[i][j];
          sumPrecip += p;
          if (p > basinMaxP) basinMaxP = p;
          if (baseTp) sumBasePrecip += baseTp[i][j];
          cellCount++;

          if (p >= 50) {
            basinExtremeCount++;
          } else if (p >= 10) {
            basinHeavyCount++;
          }
        }
      }
    }

    const meanP = cellCount > 0 ? sumPrecip / cellCount : 0;
    const meanBaseP = cellCount > 0 ? sumBasePrecip / cellCount : meanP;
    const anomalyP = meanP - meanBaseP;
    const heavyFrac = cellCount > 0 ? (basinHeavyCount + basinExtremeCount) / cellCount : 0;
    const extremeFrac = cellCount > 0 ? basinExtremeCount / cellCount : 0;

    // Normalized flood risk score (0..100):
    // - Normal July monsoon (~10-15 mm/d mean, 0 extreme) -> ~25-45 (Watch/Advisory)
    // - Active surge / deluge (Warm BoB, Western Ghats) -> ~55-85 (Warning/Critical)
    // - Widespread extreme (>= 50 mm/d) -> ~90-100
    const meanTerm = Math.min(35, meanP * 1.3);
    const heavyTerm = heavyFrac * 24;
    const peakTerm = Math.min(20, Math.max(0, basinMaxP - 10) * 0.45);
    const extremeTerm = Math.min(35, extremeFrac * 70);
    const floodScore = Math.min(
      100,
      Math.round(meanTerm + heavyTerm + peakTerm + extremeTerm)
    );

    // Crop moisture stress: high if severe drought or acute deluge
    let stress = 0;
    if (meanP < 5) {
      stress = Math.min(100, Math.round((5 - meanP) * 16)); // Drought stress
    } else if (meanP > 40) {
      stress = Math.min(100, Math.round((meanP - 40) * 2)); // Waterlogging stress
    } else {
      stress = Math.max(5, Math.round(15 + Math.abs(anomalyP) * 2));
    }

    let status: BasinImpact['status'] = 'Normal';
    if (floodScore >= 75 || stress >= 75) status = 'Critical';
    else if (floodScore >= 50 || stress >= 50) status = 'Warning';
    else if (floodScore >= 30 || stress >= 30) status = 'Advisory';
    else if (floodScore >= 15 || stress >= 15) status = 'Watch';

    return {
      regionId: b.regionId,
      name: b.name,
      latRange: b.latRange,
      lonRange: b.lonRange,
      floodRiskScore: floodScore,
      cropMoistureStress: stress,
      meanPrecipMm: Math.round(meanP * 10) / 10,
      anomalyPrecipMm: Math.round(anomalyP * 10) / 10,
      status,
    };
  });

  // Composite national indices
  const avgFlood = basins.reduce((sum, b) => sum + b.floodRiskScore, 0) / basins.length;
  const maxFlood = basins.reduce((max, b) => Math.max(max, b.floodRiskScore), 0);
  const floodRiskIndex = Math.round(0.55 * avgFlood + 0.45 * maxFlood);
  const avgStress = Math.round(
    basins.reduce((sum, b) => sum + b.cropMoistureStress, 0) / basins.length
  );

  return {
    floodRiskIndex,
    cropMoistureStressIndex: avgStress,
    extremeRainCells: extremeCount,
    heavyRainCells: heavyCount,
    drySpellCells: drySpellCount,
    highRiskAreaKm2: Math.round(extremeAreaKm2),
    basins,
  };
}
