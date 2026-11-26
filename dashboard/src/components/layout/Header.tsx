/**
 * Header.tsx - Dark slate instrument console topbar with live telemetry
 */

import React from 'react';
import { useClimateData, type ViewMode } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { Activity, Cpu, Server, Zap } from 'lucide-react';

export const Header: React.FC = () => {
  const { apiStatus, viewMode, setViewMode } = useClimateData();
  const { currentDay } = useTimeline();

  const viewModes: { id: ViewMode; label: string }[] = [
    { id: 'BASELINE', label: 'Baseline (P=0)' },
    { id: 'SHOCK', label: 'Shock Multiverse' },
    { id: 'SPLIT', label: 'Dual View' },
    { id: 'ANOMALY_DIFF', label: 'Anomaly Diff (ΔXt)' },
  ];

  return (
    <header className="h-14 border-b border-[#1E293B] bg-[#0B0F17] px-4 flex items-center justify-between select-none">
      {/* Brand & System Title */}
      <div className="flex items-center gap-3">
        <div className="w-2.5 h-2.5 rounded-full bg-sky-500 animate-pulse" />
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold tracking-wider text-slate-100 font-mono">
            MONSOONFLOW-10
          </span>
          <span className="text-xs text-slate-400 border border-slate-700 rounded px-1.5 py-0.5 font-mono">
            DIGITAL TWIN
          </span>
        </div>
      </div>

      {/* View Mode Segmented Switcher */}
      <div className="flex items-center bg-[#111827] border border-[#1E293B] rounded p-0.5">
        {viewModes.map((m) => (
          <button
            key={m.id}
            onClick={() => setViewMode(m.id)}
            className={`px-3 py-1 text-xs font-mono transition-colors rounded ${
              viewMode === m.id
                ? 'bg-[#1E293B] text-sky-400 font-medium border border-slate-700 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Telemetry Status Strip */}
      <div className="flex items-center gap-3 text-xs font-mono tabular-nums">
        {/* Forecast Lead Time */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111827] border border-[#1E293B] text-slate-300">
          <Activity className="w-3.5 h-3.5 text-sky-400" />
          <span>
            LEAD: <strong className="text-slate-100">DAY {currentDay.toString().padStart(2, '0')}</strong> ({currentDay * 24}h)
          </span>
        </div>

        {/* Offline CPU Specification Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111827] border border-[#1E293B] text-slate-400">
          <Cpu className="w-3.5 h-3.5 text-slate-400" />
          <span>4-CORE OFFLINE CPU (≤5.0s)</span>
        </div>

        {/* Inference Latency (ms) */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111827] border border-[#1E293B] text-slate-300">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span>
            {apiStatus.isLive ? 'MODEL' : 'LOCAL'} LATENCY: <strong className="text-slate-100">{apiStatus.latencyMs} ms</strong>
          </span>
        </div>

        {/* Data provenance: is what's on screen real model output? */}
        {(() => {
          const pending = apiStatus.modelName.toLowerCase().includes('pending');
          const tone = apiStatus.isLive && !pending
            ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-400'
            : apiStatus.isLive && pending
            ? 'bg-amber-950/40 border-amber-800/60 text-amber-300'
            : 'bg-rose-950/40 border-rose-800/60 text-rose-300';
          const text = apiStatus.isLive
            ? pending
              ? `LIVE :8000 [${apiStatus.modelName}] · weights pending, server is serving persistence`
              : `LIVE :8000 [${apiStatus.modelName}]`
            : `SYNTHETIC DEMO · not model output${apiStatus.fallbackReason ? ` (${apiStatus.fallbackReason})` : ''}`;
          return (
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border ${tone}`} title={text}>
              <Server className="w-3.5 h-3.5" />
              <span className="font-semibold max-w-[22rem] truncate">{text}</span>
            </div>
          );
        })()}
      </div>
    </header>
  );
};
