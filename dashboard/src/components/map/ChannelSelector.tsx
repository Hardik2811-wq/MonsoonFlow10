/**
 * ChannelSelector.tsx - Atmospheric channel selector bar
 */

import React from 'react';
import { CHANNEL_LIST } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';

export const ChannelSelector: React.FC = () => {
  const { selectedChannel, setSelectedChannel } = useClimateData();

  return (
    <div className="flex items-center gap-1 bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] rounded p-1">
      {CHANNEL_LIST.map((ch) => {
        const isSelected = selectedChannel === ch.id;
        return (
          <button
            key={ch.id}
            onClick={() => setSelectedChannel(ch.id)}
            className={`px-2.5 py-1 text-xs font-mono rounded transition-colors flex items-center gap-1.5 ${
              isSelected
                ? 'bg-white/10 text-white font-semibold border border-white/25'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <span>{ch.id.toUpperCase()}</span>
            <span className="text-[11px] text-zinc-500">({ch.displayUnit})</span>
          </button>
        );
      })}
    </div>
  );
};
