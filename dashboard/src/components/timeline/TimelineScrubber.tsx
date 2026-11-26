/**
 * TimelineScrubber.tsx - Forecast Timeline Scrubber from Day 0 to Day 10
 * Interactive temporal playback over live POST /simulate rollouts
 */

import React from 'react';
import { useTimeline } from '../../context/TimelineContext';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

export const TimelineScrubber: React.FC = () => {
  const {
    currentDay,
    isPlaying,
    playbackSpeed,
    setCurrentDay,
    togglePlay,
    setPlaybackSpeed,
    stepForward,
    stepBackward,
    resetToDay0,
  } = useTimeline();

  const days = Array.from({ length: 11 }, (_, i) => i);

  return (
    <div className="bg-[#111827] border-t border-[#1E293B] px-4 py-2.5 flex items-center justify-between gap-6 select-none font-mono text-xs">
      {/* Playback Transport Buttons */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={resetToDay0}
          className="p-1.5 rounded bg-[#1E293B] text-slate-300 hover:text-slate-100 hover:bg-slate-700 transition-colors"
          title="Reset to Day 0 (Initial Condition)"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={stepBackward}
          className="p-1.5 rounded bg-[#1E293B] text-slate-300 hover:text-slate-100 hover:bg-slate-700 transition-colors"
          title="Step Backward (-24h)"
        >
          <SkipBack className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={togglePlay}
          className="px-3.5 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-white font-medium flex items-center gap-1.5 transition-colors shadow-sm"
          title={isPlaying ? 'Pause Rollout' : 'Play Rollout (Day 0-10)'}
        >
          {isPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5" />
              <span>PAUSE</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              <span>PLAY</span>
            </>
          )}
        </button>

        <button
          onClick={stepForward}
          className="p-1.5 rounded bg-[#1E293B] text-slate-300 hover:text-slate-100 hover:bg-slate-700 transition-colors"
          title="Step Forward (+24h)"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>

        {/* Speed Switcher */}
        <div className="ml-2 flex items-center bg-[#0B0F17] border border-[#1E293B] rounded p-0.5">
          {[0.5, 1, 2].map((s) => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                playbackSpeed === s
                  ? 'bg-[#1E293B] text-sky-400 font-semibold border border-slate-700'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Interactive Range Scrubber with Keyframe Nodes */}
      <div className="flex-1 flex flex-col justify-center px-4">
        <div className="flex justify-between text-[11px] text-slate-400 mb-1 tabular-nums">
          <span>DAY 00 (Initial Condition)</span>
          <span className="font-semibold text-sky-400">
            DAY {currentDay.toString().padStart(2, '0')} ({currentDay * 24} HOURS LEAD)
          </span>
          <span>DAY 10 (+240h Rollout)</span>
        </div>

        {/* Track with active fill */}
        <div className="relative w-full flex items-center">
          <input
            type="range"
            min={0}
            max={10}
            step={1}
            value={currentDay}
            onChange={(e) => setCurrentDay(parseInt(e.target.value, 10))}
            className="w-full h-2 bg-[#1E293B] rounded-lg appearance-none cursor-pointer z-10 accent-sky-500"
          />
        </div>

        {/* Discrete Keyframe Ticks with Interactive Clicking */}
        <div className="flex justify-between mt-1 px-1">
          {days.map((d) => (
            <button
              key={d}
              onClick={() => setCurrentDay(d)}
              className={`text-[10px] tabular-nums transition-colors px-1 py-0.5 rounded ${
                currentDay === d
                  ? 'text-sky-400 font-bold bg-sky-950/50 border border-sky-800/60'
                  : 'text-slate-600 hover:text-slate-400'
              }`}
            >
              D{d}
            </button>
          ))}
        </div>
      </div>

      {/* Lead Time Display Pill */}
      <div className="px-3 py-1.5 rounded bg-[#0B0F17] border border-[#1E293B] text-slate-300 text-center min-w-[125px]">
        <div className="text-[10px] text-slate-500">LEAD HORIZON</div>
        <div className="font-semibold text-sky-400 tabular-nums">
          +{currentDay * 24}h FORECAST
        </div>
      </div>
    </div>
  );
};
