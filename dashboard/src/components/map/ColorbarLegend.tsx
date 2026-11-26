/**
 * ColorbarLegend.tsx - Quantitative meteorological color scale indicator
 */

import React from 'react';
import { CHANNELS } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { getColorForChannel, getAnomalyDivergentColor, rgbToCss } from '../../services/colormaps';

interface Props {
  isAnomaly?: boolean;
  maxAbsDelta?: number;
}

export const ColorbarLegend: React.FC<Props> = ({ isAnomaly = false, maxAbsDelta = 5 }) => {
  const { selectedChannel } = useClimateData();
  const meta = CHANNELS[selectedChannel];

  // Number of color gradient samples
  const steps = 24;
  const samples: string[] = [];

  if (isAnomaly) {
    for (let s = 0; s <= steps; s++) {
      const delta = -maxAbsDelta + (2 * maxAbsDelta * s) / steps;
      const rgb = getAnomalyDivergentColor(delta, maxAbsDelta);
      samples.push(rgbToCss(rgb));
    }
  } else {
    for (let s = 0; s <= steps; s++) {
      const val = meta.min + ((meta.max - meta.min) * s) / steps;
      const rgb = getColorForChannel(val, selectedChannel);
      samples.push(rgbToCss(rgb));
    }
  }

  const gradientCss = `linear-gradient(to right, ${samples.join(', ')})`;

  return (
    <div className="bg-[#111827]/90 backdrop-blur-sm border border-[#1E293B] rounded p-2.5 shadow-lg select-none text-xs font-mono">
      <div className="flex justify-between items-center mb-1 text-slate-300">
        <span className="font-semibold text-slate-100">
          {isAnomaly ? `ANOMALY Δ ${meta.id.toUpperCase()}` : meta.label}
        </span>
        <span className="text-slate-400">
          {isAnomaly ? `Δ ${meta.displayUnit}` : meta.displayUnit}
        </span>
      </div>

      {/* Color gradient strip */}
      <div
        className="h-3 w-64 rounded border border-slate-700/80 mb-1"
        style={{ background: gradientCss }}
      />

      {/* Numeric ticks */}
      <div className="flex justify-between text-[11px] text-slate-400 tabular-nums">
        {isAnomaly ? (
          <>
            <span>-{maxAbsDelta.toFixed(1)}</span>
            <span>0.0</span>
            <span>+{maxAbsDelta.toFixed(1)}</span>
          </>
        ) : (
          <>
            <span>
              {selectedChannel === 't2m'
                ? `${(meta.min - 273.15).toFixed(0)}°`
                : selectedChannel === 'msl'
                ? `${(meta.min / 100).toFixed(0)}`
                : meta.min}
            </span>
            <span>
              {selectedChannel === 't2m'
                ? `${(((meta.min + meta.max) / 2) - 273.15).toFixed(0)}°`
                : selectedChannel === 'msl'
                ? `${(((meta.min + meta.max) / 2) / 100).toFixed(0)}`
                : ((meta.min + meta.max) / 2).toFixed(0)}
            </span>
            <span>
              {selectedChannel === 't2m'
                ? `${(meta.max - 273.15).toFixed(0)}°`
                : selectedChannel === 'msl'
                ? `${(meta.max / 100).toFixed(0)}`
                : meta.max}
            </span>
          </>
        )}
      </div>
    </div>
  );
};
