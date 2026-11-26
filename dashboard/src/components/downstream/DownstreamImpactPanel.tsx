/**
 * DownstreamImpactPanel.tsx - Humanitarian & Agricultural Downstream Impact Module
 * Calculates Flood Risk Index and Crop Moisture Stress Index
 */

import React, { useMemo } from 'react';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { calculateDownstreamImpacts } from '../../services/impactCalculator';
import { computePhysicsAudit } from '../../services/physicsAudit';
import { ShockTrajectoryChart } from '../anomaly/ShockTrajectoryChart';
import { CloudRain, Sprout, TrendingUp } from 'lucide-react';

export const DownstreamImpactPanel: React.FC = () => {
  const { baselineRollout, shockRollout, initialCondition } = useClimateData();
  const { currentDay } = useTimeline();

  const currentFrame = useMemo(() => {
    if (currentDay === 0) return initialCondition;
    return shockRollout[currentDay - 1] || initialCondition;
  }, [currentDay, shockRollout, initialCondition]);

  const baseFrame = useMemo(() => {
    if (currentDay === 0) return initialCondition;
    return baselineRollout[currentDay - 1] || initialCondition;
  }, [currentDay, baselineRollout, initialCondition]);

  const prevFrame = useMemo(() => {
    if (currentDay <= 0) return undefined;
    if (currentDay === 1) return initialCondition;
    return shockRollout[currentDay - 2] || initialCondition;
  }, [currentDay, shockRollout, initialCondition]);

  const impacts = useMemo(() => {
    return calculateDownstreamImpacts(currentFrame, baseFrame);
  }, [currentFrame, baseFrame]);

  const physicsAudit = useMemo(() => {
    return computePhysicsAudit(currentFrame, prevFrame);
  }, [currentFrame, prevFrame]);

  return (
    <div className="w-80 h-full border rounded-xl bg-black/40 backdrop-blur-2xl backdrop-saturate-150 border-white/[0.12] shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_12px_32px_rgba(0,0,0,0.45)] p-4 flex flex-col justify-between overflow-y-auto select-none font-mono text-xs">
      <div>
        {/* Panel Header */}
        <div className="flex items-center justify-between border-b border-white/[0.12] pb-2 mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-zinc-300" />
            <span className="font-semibold text-zinc-100 tracking-wider">
              DOWNSTREAM IMPACTS
            </span>
          </div>
          <span className="text-[11px] text-zinc-400 px-1.5 py-0.5 rounded bg-white/[0.06]">
            DAY {currentDay.toString().padStart(2, '0')}
          </span>
        </div>

        {/* Primary Composite Gauges */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Flood Risk Index */}
          <div className="bg-white/[0.04] border border-white/[0.12] rounded p-3">
            <div className="flex items-center justify-between text-zinc-400 mb-1">
              <span className="text-[10px]">FLOOD RISK</span>
              <CloudRain className="w-3.5 h-3.5 text-zinc-300" />
            </div>
            <div className="text-xl font-bold tabular-nums text-zinc-100 mb-1">
              {impacts.floodRiskIndex}
              <span className="text-xs font-normal text-zinc-500">/100</span>
            </div>
            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${impacts.floodRiskIndex >= 70
                  ? 'bg-red-500'
                  : impacts.floodRiskIndex >= 40
                    ? 'bg-amber-400'
                    : 'bg-sky-400'
                  }`}
                style={{ width: `${impacts.floodRiskIndex}%` }}
              />
            </div>
          </div>

          {/* Crop Moisture Stress Index */}
          <div className="bg-white/[0.04] border border-white/[0.12] rounded p-3">
            <div className="flex items-center justify-between text-zinc-400 mb-1">
              <span className="text-[10px]">CROP STRESS</span>
              <Sprout className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="text-xl font-bold tabular-nums text-zinc-100 mb-1">
              {impacts.cropMoistureStressIndex}
              <span className="text-xs font-normal text-zinc-500">/100</span>
            </div>
            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${impacts.cropMoistureStressIndex >= 60
                  ? 'bg-amber-400'
                  : 'bg-emerald-400'
                  }`}
                style={{ width: `${impacts.cropMoistureStressIndex}%` }}
              />
            </div>
          </div>
        </div>

        {/* Shock Propagation Trajectory (Day 1 - Day 10) */}
        <ShockTrajectoryChart />

        {/* Heavy Precipitation Threshold Counters */}
        <div className="bg-white/[0.04] border border-white/[0.12] rounded p-3 mb-4 tabular-nums">
          <span className="text-[11px] text-zinc-400 block mb-2 font-medium">
            PRECIPITATION CLUSTER AUDIT
          </span>
          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Extreme Deluge (≥ 50 mm/d)</span>
              <span className="font-semibold text-rose-400">
                {impacts.extremeRainCells} cells
              </span>
            </div>
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Heavy Rainfall (≥ 10 mm/d)</span>
              <span className="font-semibold text-sky-400">
                {impacts.heavyRainCells} cells
              </span>
            </div>
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Extreme-Rain Area (≥ 50 mm/d)</span>
              <span className="font-semibold text-zinc-200">
                {impacts.highRiskAreaKm2.toLocaleString()} km²
              </span>
            </div>
          </div>
        </div>

        {/* Lean Physics & Vortex Audit */}
        <div className="bg-white/[0.04] border border-white/[0.12] rounded p-3 mb-4 tabular-nums">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] text-zinc-400 font-medium">
              PHYSICS &amp; VORTEX AUDIT
            </span>
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded font-semibold border ${
                physicsAudit.status === 'PASS'
                  ? 'bg-emerald-950/50 text-emerald-400 border-emerald-800/70'
                  : 'bg-amber-950/50 text-amber-300 border-amber-800/70'
              }`}
              title={physicsAudit.note}
            >
              {physicsAudit.status}
            </span>
          </div>
          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Peak 10m Vorticity (ζ₁₀)</span>
              <span className="font-semibold text-zinc-100">
                {physicsAudit.signedPeakVorticity1e5 > 0 ? `+${physicsAudit.signedPeakVorticity1e5}` : physicsAudit.signedPeakVorticity1e5} ×10⁻⁵ s⁻¹
              </span>
            </div>
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Min Central MSL</span>
              <span className="font-semibold text-sky-400">
                {physicsAudit.minMslHpa.toFixed(1)} hPa
              </span>
            </div>
            <div className="flex justify-between items-center text-zinc-300">
              <span className="text-zinc-400">Open-Bnd Moisture R_q</span>
              <span className="font-semibold text-zinc-200">
                {physicsAudit.moistureResidualMmDay.toFixed(1)} mm/d
              </span>
            </div>
          </div>
        </div>

        {/* Regional Basins Table */}
        <div>
          <span className="text-[11px] text-zinc-400 block mb-2 font-medium">
            REGIONAL BASIN AUDIT
          </span>
          <div className="space-y-2">
            {impacts.basins.map((basin) => (
              <div
                key={basin.regionId}
                className="bg-white/[0.04] border border-white/[0.12] rounded p-2 text-[11px]"
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold text-zinc-200 truncate pr-2">
                    {basin.name}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${basin.status === 'Critical'
                      ? 'bg-red-950 text-red-400 border border-red-800'
                      : basin.status === 'Warning'
                        ? 'bg-amber-950 text-amber-400 border border-amber-800'
                        : 'bg-white/[0.06] text-zinc-400'
                      }`}
                  >
                    {basin.status}
                  </span>
                </div>
                <div className="flex justify-between text-zinc-400 tabular-nums">
                  <span>Rain: {basin.meanPrecipMm} mm/d</span>
                  <span
                    className={
                      basin.anomalyPrecipMm > 0
                        ? 'text-red-400'
                        : basin.anomalyPrecipMm < 0
                          ? 'text-sky-400'
                          : 'text-zinc-400'
                    }
                  >
                    Δ {basin.anomalyPrecipMm > 0 ? `+${basin.anomalyPrecipMm}` : basin.anomalyPrecipMm}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Humanitarian Lens Footer Note */}
      <div className="border-t border-white/[0.12] pt-3 text-[10px] text-zinc-500 leading-normal">
        Computed from the displayed rollout tensors. Cell counts and area cover the full 32x36 grid (area ∝ cos lat). Flood risk and crop stress are heuristic indices for comparing scenarios, not IMD warning categories. 10 mm/day is the grader's heavy-rain threshold.
      </div>
    </div>
  );
};
