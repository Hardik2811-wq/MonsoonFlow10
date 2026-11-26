/**
 * ChannelSelector.tsx - Atmospheric channel selector bar
 */

import React from 'react';
import { CHANNEL_LIST } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';

export const ChannelSelector: React.FC = () => {
  const { selectedChannel, setSelectedChannel } = useClimateData();

  return (
    <div className="flex items-center gap-1 bg-[#111827] border border-[#1E293B] rounded p-1">
      {CHANNEL_LIST.map((ch) => {
        const isSelected = selectedChannel === ch.id;
        return (
          <button
            key={ch.id}
            onClick={() => setSelectedChannel(ch.id)}
            className={`px-2.5 py-1 text-xs font-mono rounded transition-colors flex items-center gap-1.5 ${
              isSelected
                ? 'bg-[#1E293B] text-sky-400 font-semibold border border-slate-700'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span>{ch.id.toUpperCase()}</span>
            <span className="text-[11px] text-slate-500">({ch.displayUnit})</span>
          </button>
        );
      })}
    </div>
  );
};
