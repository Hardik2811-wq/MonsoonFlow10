/**
 * ShockLabPanel.tsx - Multiverse Counterfactual Perturbation Laboratory
 */

import React from 'react';
import { AMPLITUDE_RANGE, SHOCK_PRESETS } from '../../types/shocks';
import { useShockLab } from '../../context/ShockLabContext';
import { CHANNEL_LIST, CHANNELS, type ChannelId } from '../../types/climate';
import { AlertTriangle, CheckCircle, Paintbrush, RefreshCw, Sliders } from 'lucide-react';

export const ShockLabPanel: React.FC = () => {
  const {
    activePresetId,
    customSpec,
    isBrushMode,
    boundsReport,
    selectPreset,
    updateCustomSpec,
    toggleBrushMode,
    resetShock,
  } = useShockLab();

  return (
    <div className="w-80 h-full border rounded-xl bg-black/40 backdrop-blur-2xl backdrop-saturate-150 border-white/[0.12] shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_12px_32px_rgba(0,0,0,0.45)] p-4 flex flex-col justify-between overflow-y-auto select-none font-mono text-xs">
      <div>
        {/* Panel Header */}
        <div className="flex items-center justify-between border-b border-white/[0.12] pb-2 mb-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-zinc-300" />
            <span className="font-semibold text-zinc-100 tracking-wider">
              SHOCK LAB
            </span>
          </div>
          <button
            onClick={resetShock}
            className="text-[11px] text-zinc-400 hover:text-zinc-200 flex items-center gap-1 hover:underline"
          >
            <RefreshCw className="w-3 h-3" />
            <span>RESET (P=0)</span>
          </button>
        </div>

        {/* Counterfactual Presets */}
        <div className="mb-5">
          <span className="text-zinc-400 block mb-2 font-medium">PRESET SHOCKS</span>
          <div className="space-y-1.5">
            {SHOCK_PRESETS.map((preset) => {
              const isSelected = activePresetId === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => selectPreset(preset.id)}
                  className={`w-full text-left p-2 rounded border transition-colors ${isSelected
                    ? 'bg-white/10 border-white/25 text-white'
                    : 'bg-white/[0.04] border-white/[0.12] text-zinc-300 hover:border-white/20'
                    }`}
                >
                  <div className="flex justify-between items-center mb-0.5">
                    <span className="font-semibold text-zinc-100">{preset.name}</span>
                    <span className="flex gap-1">
                      {preset.graded && (
                        <span className="text-[10px] px-1 rounded bg-white/[0.06] text-zinc-200 border border-white/20">
                          GRADED
                        </span>
                      )}
                      <span className="text-[10px] px-1 rounded bg-white/[0.06] text-zinc-400">
                        {preset.category}
                      </span>
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-400 line-clamp-2 leading-relaxed">
                    {preset.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Custom 2D Gaussian Shock Brush */}
        <div className="mb-5 border-t border-white/[0.12] pt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-zinc-400 font-medium">
              INTERACTIVE GAUSSIAN BRUSH
              {activePresetId === 'CUSTOM' && <span className="ml-1 text-white">· ACTIVE</span>}
            </span>
            <button
              onClick={toggleBrushMode}
              className={`px-2 py-1 rounded border flex items-center gap-1 transition-colors ${isBrushMode
                ? 'bg-white/10 border-white/25 text-white font-medium'
                : 'bg-white/[0.04] border-white/[0.12] text-zinc-400 hover:text-zinc-200'
                }`}
            >
              <Paintbrush className="w-3 h-3" />
              <span>{isBrushMode ? 'BRUSH ACTIVE' : 'PAINT SHOCK'}</span>
            </button>
          </div>

          <div className="space-y-3 bg-white/[0.04] p-2.5 rounded border border-white/[0.12] tabular-nums">
            {/* Target Channel */}
            <div>
              <label className="text-[11px] text-zinc-400 block mb-1">
                TARGET CHANNEL
              </label>
              <select
                value={customSpec.channel}
                onChange={(e) => {
                  const channel = e.target.value as ChannelId;
                  const r = AMPLITUDE_RANGE[channel];
                  // keep the amplitude inside the new channel's range
                  updateCustomSpec({ channel, magnitude: Math.max(r.min, Math.min(r.max, customSpec.magnitude)) });
                }}
                className="w-full bg-black/30 border border-white/[0.12] rounded p-1 text-zinc-200 font-mono text-xs"
              >
                {CHANNEL_LIST.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id.toUpperCase()} - {c.label} ({c.wireUnit})
                  </option>
                ))}
              </select>
            </div>

            {/* Shock Center (Lat, Lon) */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-zinc-400 block mb-0.5">
                  CENTER LAT
                </label>
                <input
                  type="number"
                  step="0.5"
                  min="6.5"
                  max="37.5"
                  value={customSpec.centerLat}
                  onChange={(e) => updateCustomSpec({ centerLat: parseFloat(e.target.value) || 20 })}
                  className="w-full bg-black/30 border border-white/[0.12] rounded p-1 text-zinc-200 font-mono"
                />
              </div>
              <div>
                <label className="text-[11px] text-zinc-400 block mb-0.5">
                  CENTER LON
                </label>
                <input
                  type="number"
                  step="0.5"
                  min="65.5"
                  max="100.5"
                  value={customSpec.centerLon}
                  onChange={(e) => updateCustomSpec({ centerLon: parseFloat(e.target.value) || 80 })}
                  className="w-full bg-black/30 border border-white/[0.12] rounded p-1 text-zinc-200 font-mono"
                />
              </div>
            </div>

            {/* Perturbation Magnitude */}
            <div>
              <div className="flex justify-between text-[11px] text-zinc-400 mb-0.5">
                <span>AMPLITUDE (Δ)</span>
                <span className="text-white font-semibold">
                  {customSpec.magnitude > 0 ? `+${customSpec.magnitude}` : customSpec.magnitude} {CHANNELS[customSpec.channel].wireUnit}
                </span>
              </div>
              <input
                type="range"
                min={AMPLITUDE_RANGE[customSpec.channel].min}
                max={AMPLITUDE_RANGE[customSpec.channel].max}
                step={AMPLITUDE_RANGE[customSpec.channel].step}
                value={customSpec.magnitude}
                onChange={(e) => updateCustomSpec({ magnitude: parseFloat(e.target.value) })}
                className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer"
              />
            </div>

            {/* Gaussian Radius & Sigma */}
            <div>
              <div className="flex justify-between text-[11px] text-zinc-400 mb-0.5">
                <span>GAUSSIAN SIGMA (σ)</span>
                <span className="text-zinc-200">{customSpec.sigmaKm} km</span>
              </div>
              <input
                type="range"
                min="100"
                max="600"
                step="25"
                value={customSpec.sigmaKm}
                onChange={(e) => updateCustomSpec({ sigmaKm: parseInt(e.target.value, 10) })}
                className="w-full h-1 bg-white/10 rounded appearance-none cursor-pointer"
              />
              <span className="text-[10px] text-zinc-500 mt-0.5 block">
                σ ≈ 300 km (~2.5 grid cells) prevents high-frequency noise spikes.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Physical Realism & Causality Validation Gate */}
      <div className="border-t border-white/[0.12] pt-3">
        {boundsReport.isValid ? (
          <div className="flex items-center gap-2 p-2 rounded bg-emerald-950/30 border border-emerald-800/40 text-emerald-400">
            <CheckCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-[11px]">
              Within Physical Bounds: Perturbation preserves atmospheric stability.
            </span>
          </div>
        ) : (
          <div className="p-2 rounded bg-amber-950/40 border border-amber-800/50 text-amber-300">
            <div className="flex items-center gap-1.5 font-semibold text-[11px] mb-1">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
              <span>PHYSICAL BOUNDS ADVISORY</span>
            </div>
            {boundsReport.warnings.map((w, idx) => (
              <p key={idx} className="text-[10px] text-amber-400/90 leading-tight">
                • {w}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
