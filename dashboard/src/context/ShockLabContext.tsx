/**
 * ShockLabContext.tsx - Counterfactual perturbation configuration state
 */

import React, { createContext, useContext, useState, useMemo, useCallback } from 'react';
import type { GaussianShockSpec, PresetShockId, ShockPreset } from '../types/shocks';
import { SHOCK_PRESETS } from '../types/shocks';
import { type ShockBoundsReport, validateShockRealism } from '../services/gaussianBrush';

interface ShockLabContextType {
  activePresetId: PresetShockId;
  activePreset: ShockPreset;
  customSpec: GaussianShockSpec;
  isBrushMode: boolean;
  boundsReport: ShockBoundsReport;
  selectPreset: (presetId: PresetShockId) => void;
  updateCustomSpec: (partial: Partial<GaussianShockSpec>) => void;
  toggleBrushMode: () => void;
  setBrushMode: (active: boolean) => void;
  resetShock: () => void;
}

const DEFAULT_CUSTOM_SPEC: GaussianShockSpec = {
  centerLat: 15.0,
  centerLon: 88.0,
  radiusKm: 450,
  sigmaKm: 300,
  channel: 't2m',
  magnitude: 3.0,
};

const ShockLabContext = createContext<ShockLabContextType | undefined>(undefined);

export const ShockLabProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activePresetId, setActivePresetId] = useState<PresetShockId>('ZERO_SHOCK');
  const [customSpec, setCustomSpec] = useState<GaussianShockSpec>(DEFAULT_CUSTOM_SPEC);
  const [isBrushMode, setIsBrushMode] = useState<boolean>(false);

  const activePreset = useMemo(
    () => SHOCK_PRESETS.find((p) => p.id === activePresetId) || SHOCK_PRESETS[3],
    [activePresetId]
  );

  const selectPreset = useCallback((presetId: PresetShockId) => {
    setActivePresetId(presetId);
    setIsBrushMode(false);
    const preset = SHOCK_PRESETS.find((p) => p.id === presetId);
    if (preset?.spec) {
      setCustomSpec({ ...preset.spec });
    }
  }, []);

  const updateCustomSpec = useCallback((partial: Partial<GaussianShockSpec>) => {
    setCustomSpec((prev) => ({ ...prev, ...partial }));
    // If user modifies custom parameters, deselect preset if not matching
    setActivePresetId((prev) => {
      const preset = SHOCK_PRESETS.find((p) => p.id === prev);
      return preset?.spec ? 'ZERO_SHOCK' : prev;
    });
  }, []);

  const toggleBrushMode = useCallback(() => {
    setIsBrushMode((prev) => !prev);
  }, []);

  const setBrushMode = useCallback((active: boolean) => {
    setIsBrushMode(active);
  }, []);

  const resetShock = useCallback(() => {
    setActivePresetId('ZERO_SHOCK');
    setIsBrushMode(false);
    setCustomSpec(DEFAULT_CUSTOM_SPEC);
  }, []);

  const boundsReport = useMemo(
    () => validateShockRealism(customSpec.channel, customSpec.magnitude),
    [customSpec.channel, customSpec.magnitude]
  );

  return (
    <ShockLabContext.Provider
      value={{
        activePresetId,
        activePreset,
        customSpec,
        isBrushMode,
        boundsReport,
        selectPreset,
        updateCustomSpec,
        toggleBrushMode,
        setBrushMode,
        resetShock,
      }}
    >
      {children}
    </ShockLabContext.Provider>
  );
};

export function useShockLab(): ShockLabContextType {
  const ctx = useContext(ShockLabContext);
  if (!ctx) throw new Error('useShockLab must be used within ShockLabProvider');
  return ctx;
}
