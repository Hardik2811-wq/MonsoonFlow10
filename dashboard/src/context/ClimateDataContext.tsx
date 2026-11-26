/**
 * ClimateDataContext.tsx - Master atmospheric simulation state coordinator
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import type { ChannelId, ChannelTensor, SimulationRollout } from '../types/climate';
import type { ApiStatus } from '../types/api';
import { checkBackendHealth, generate3DayClimatologicalHistory, runSimulation } from '../services/api';
import { useShockLab } from './ShockLabContext';

export type ViewMode = 'BASELINE' | 'SHOCK' | 'SPLIT' | 'ANOMALY_DIFF';

interface ClimateDataContextType {
  initialCondition: ChannelTensor; // Day 0 state [6, 32, 36]
  history3Day: ChannelTensor[];    // 3-day history [3, 6, 32, 36]
  baselineRollout: SimulationRollout;
  shockRollout: SimulationRollout;
  selectedChannel: ChannelId;
  viewMode: ViewMode;
  apiStatus: ApiStatus;
  setSelectedChannel: (channel: ChannelId) => void;
  setViewMode: (mode: ViewMode) => void;
  executeSimulation: () => Promise<void>;
}

const ClimateDataContext = createContext<ClimateDataContextType | undefined>(undefined);

export const ClimateDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { activePresetId, customSpec } = useShockLab();

  const [history3Day] = useState<ChannelTensor[]>(() => generate3DayClimatologicalHistory());
  const initialCondition = history3Day[2]; // Day 0 state

  const [baselineRollout, setBaselineRollout] = useState<SimulationRollout>([]);
  const [shockRollout, setShockRollout] = useState<SimulationRollout>([]);
  const [selectedChannel, setSelectedChannel] = useState<ChannelId>('tp'); // Default to precipitation
  const [viewMode, setViewMode] = useState<ViewMode>('SHOCK');

  const [apiStatus, setApiStatus] = useState<ApiStatus>({
    isLive: false,
    isSimulating: false,
    latencyMs: 0,
    endpoint: 'http://localhost:8000',
    modelName: 'MonsoonFlow-10',
    lastSimulatedAt: null,
    errorMessage: null,
  });

  // Periodic health check to detect live server status
  useEffect(() => {
    let isMounted = true;
    const pollHealth = async () => {
      const { isOnline, info } = await checkBackendHealth();
      if (isMounted) {
        setApiStatus((prev) => ({
          ...prev,
          isLive: isOnline,
          modelName: info?.model_name || prev.modelName,
        }));
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Compute baseline rollout once on initial mount
  useEffect(() => {
    let isMounted = true;
    const initBaseline = async () => {
      const res = await runSimulation(history3Day, null);
      if (isMounted) {
        setBaselineRollout(res.rollout);
        setShockRollout(res.rollout);
        setApiStatus((prev) => ({
          ...prev,
          isLive: res.isLiveBackend,
          latencyMs: res.inferenceTimeMs,
          lastSimulatedAt: new Date(),
        }));
      }
    };
    initBaseline();
  }, [history3Day]);

  // Execute shock simulation whenever shock spec or preset changes
  const executeSimulation = useCallback(async () => {
    setApiStatus((prev) => ({ ...prev, isSimulating: true }));
    const activeSpec = activePresetId === 'ZERO_SHOCK' ? null : customSpec;

    try {
      const res = await runSimulation(history3Day, activeSpec);
      setShockRollout(res.rollout);
      setApiStatus((prev) => ({
        ...prev,
        isSimulating: false,
        isLive: res.isLiveBackend,
        latencyMs: res.inferenceTimeMs,
        lastSimulatedAt: new Date(),
        errorMessage: null,
      }));
    } catch (err: unknown) {
      setApiStatus((prev) => ({
        ...prev,
        isSimulating: false,
        errorMessage: err instanceof Error ? err.message : 'Simulation failed',
      }));
    }
  }, [history3Day, activePresetId, customSpec]);

  // Auto re-run simulation on shock spec change
  useEffect(() => {
    if (baselineRollout.length > 0) {
      executeSimulation();
    }
  }, [activePresetId, customSpec, executeSimulation]);

  return (
    <ClimateDataContext.Provider
      value={{
        initialCondition,
        history3Day,
        baselineRollout,
        shockRollout,
        selectedChannel,
        viewMode,
        apiStatus,
        setSelectedChannel,
        setViewMode,
        executeSimulation,
      }}
    >
      {children}
    </ClimateDataContext.Provider>
  );
};

export function useClimateData(): ClimateDataContextType {
  const ctx = useContext(ClimateDataContext);
  if (!ctx) throw new Error('useClimateData must be used within ClimateDataProvider');
  return ctx;
}
