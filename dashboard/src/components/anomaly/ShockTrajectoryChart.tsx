/**
 * ShockTrajectoryChart.tsx - Day 1 to Day 10 Shock Propagation Trajectory (Recharts)
 * Visualizes temporal evolution of Delta t2m (K) and Delta tp (mm/day)
 * computed dynamically from live baselineRollout and shockRollout tensors.
 */

import React, { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { GRID_CONFIG } from '../../types/climate';
import { Activity } from 'lucide-react';

interface TrajectoryPoint {
  day: number;
  label: string;
  peakT2m: number;
  meanT2m: number;
  peakTp: number;
  meanTp: number;
  spreadCells: number;
}

export const ShockTrajectoryChart: React.FC = () => {
  const { baselineRollout, shockRollout } = useClimateData();
  const { currentDay, setCurrentDay } = useTimeline();
  const [metricMode, setMetricMode] = useState<'PEAK' | 'MEAN'>('PEAK');

  const data: TrajectoryPoint[] = useMemo(() => {
    const steps = Math.min(baselineRollout.length, shockRollout.length);
    if (steps === 0) return [];

    const totalCells = GRID_CONFIG.nLat * GRID_CONFIG.nLon;
    const points: TrajectoryPoint[] = [];

    for (let t = 0; t < steps; t++) {
      const baseFrame = baselineRollout[t];
      const shockFrame = shockRollout[t];
      if (!baseFrame || !shockFrame) continue;

      const baseT2m = baseFrame[0];
      const shockT2m = shockFrame[0];
      const baseTp = baseFrame[1];
      const shockTp = shockFrame[1];

      let sumDeltaT2m = 0;
      let maxAbsDeltaT2m = 0;
      let sumDeltaTp = 0;
      let maxAbsDeltaTp = 0;
      let spreadCount = 0;

      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          const dT2m = shockT2m[i][j] - baseT2m[i][j];
          const dTp = shockTp[i][j] - baseTp[i][j];

          sumDeltaT2m += dT2m;
          sumDeltaTp += dTp;

          const absT2m = Math.abs(dT2m);
          const absTp = Math.abs(dTp);

          if (absT2m > maxAbsDeltaT2m) maxAbsDeltaT2m = absT2m;
          if (absTp > maxAbsDeltaTp) maxAbsDeltaTp = absTp;

          if (absT2m >= 0.1 || absTp >= 0.25) {
            spreadCount++;
          }
        }
      }

      points.push({
        day: t + 1,
        label: `D${t + 1}`,
        peakT2m: Math.round(maxAbsDeltaT2m * 100) / 100,
        meanT2m: Math.round((sumDeltaT2m / totalCells) * 1000) / 1000,
        peakTp: Math.round(maxAbsDeltaTp * 100) / 100,
        meanTp: Math.round((sumDeltaTp / totalCells) * 1000) / 1000,
        spreadCells: spreadCount,
      });
    }

    return points;
  }, [baselineRollout, shockRollout]);

  const t2mKey = metricMode === 'PEAK' ? 'peakT2m' : 'meanT2m';
  const tpKey = metricMode === 'PEAK' ? 'peakTp' : 'meanTp';

  return (
    <div className="bg-[#0B0F17] border border-[#1E293B] rounded p-3 mb-4 font-mono">
      {/* Header & Mode Toggle */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-[11px] font-medium text-slate-200">
            SHOCK TRAJECTORY (D1–D10)
          </span>
        </div>

        <div className="flex items-center bg-[#111827] border border-[#1E293B] rounded p-0.5 text-[10px]">
          <button
            onClick={() => setMetricMode('PEAK')}
            className={`px-1.5 py-0.5 rounded transition-colors ${
              metricMode === 'PEAK'
                ? 'bg-[#1E293B] text-sky-400 font-semibold'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            PEAK |Δ|
          </button>
          <button
            onClick={() => setMetricMode('MEAN')}
            className={`px-1.5 py-0.5 rounded transition-colors ${
              metricMode === 'MEAN'
                ? 'bg-[#1E293B] text-sky-400 font-semibold'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            MEAN Δ
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between text-[10px] text-slate-400 mb-2 tabular-nums">
        <span className="flex items-center gap-1">
          <span className="w-2 h-0.5 bg-[#EF4444] inline-block" />
          <span>Δ t2m (K)</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-0.5 bg-[#38BDF8] inline-block" />
          <span>Δ tp (mm/d)</span>
        </span>
      </div>

      {/* Recharts LineChart */}
      <div className="h-36 w-full">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 4, right: 4, left: -22, bottom: 0 }}
              onClick={(state) => {
                if (state && state.activeLabel) {
                  const dayNum = parseInt(String(state.activeLabel).replace('D', ''), 10);
                  if (!Number.isNaN(dayNum)) setCurrentDay(dayNum);
                }
              }}
            >
              <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: '#64748B', fontSize: 10 }}
                axisLine={{ stroke: '#1E293B' }}
                tickLine={false}
              />
              <YAxis
                yAxisId="t2m"
                tick={{ fill: '#EF4444', fontSize: 9 }}
                axisLine={false}
                tickLine={false}
                width={38}
              />
              <YAxis
                yAxisId="tp"
                orientation="right"
                tick={{ fill: '#38BDF8', fontSize: 9 }}
                axisLine={false}
                tickLine={false}
                width={34}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#111827',
                  borderColor: '#334155',
                  borderRadius: '4px',
                  fontSize: '10px',
                  fontFamily: 'monospace',
                  color: '#F1F5F9',
                }}
                formatter={(value, name) => {
                  const num = Number(value);
                  if (name === t2mKey) return [`${num.toFixed(2)} K`, 'Δ t2m'];
                  if (name === tpKey) return [`${num.toFixed(2)} mm/d`, 'Δ tp'];
                  return [String(value), String(name)];
                }}
                labelFormatter={(label, payload) => {
                  const pt = payload?.[0]?.payload as TrajectoryPoint | undefined;
                  return pt ? `${label} | Spread: ${pt.spreadCells} cells` : String(label);
                }}
              />
              {currentDay >= 1 && currentDay <= 10 && (
                <ReferenceLine
                  yAxisId="t2m"
                  x={`D${currentDay}`}
                  stroke="#94A3B8"
                  strokeDasharray="2 2"
                />
              )}
              <Line
                yAxisId="t2m"
                type="monotone"
                dataKey={t2mKey}
                stroke="#EF4444"
                strokeWidth={1.75}
                dot={{ r: 2, fill: '#EF4444' }}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
              <Line
                yAxisId="tp"
                type="monotone"
                dataKey={tpKey}
                stroke="#38BDF8"
                strokeWidth={1.75}
                dot={{ r: 2, fill: '#38BDF8' }}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-[10px] text-slate-600">
            Awaiting rollout tensor...
          </div>
        )}
      </div>
    </div>
  );
};
