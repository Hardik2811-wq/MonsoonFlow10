/**
 * ClimateDataContext.tsx - Master atmospheric simulation state coordinator
 *
 * - Day 0 comes from public/data/initial_state.json (real validation data), with its true doy_sincos.
 * - Baseline and shock rollouts are always computed together from the same source, so the
 *   Anomaly Diff never mixes a live rollout with a synthetic one.
 * - Both are recomputed whenever the shock changes or the server goes online/offline.
 */

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ChannelId, ChannelTensor, SimulationRollout } from '../types/climate';
import type { ApiStatus } from '../types/api';
import { checkBackendHealth, runSimulation, type SimulationResult } from '../services/api';
import { loadInitialState, type InitialState } from '../services/initialState';
import { resolvePerturbation } from '../services/perturbation';
import { createZeroTensor, simulateOfflineRollout } from '../services/syntheticData';
import { useShockLab } from './ShockLabContext';

export type ViewMode = 'BASELINE' | 'SHOCK' | 'SPLIT' | 'ANOMALY_DIFF';

interface ClimateDataContextType {
  initialCondition: ChannelTensor; // Day 0 state [6, 32, 36]
  history3Day: ChannelTensor[]; // 3-day history [3, 6, 32, 36]
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

const PLACEHOLDER_HISTORY: ChannelTensor[] = [createZeroTensor(), createZeroTensor(), createZeroTensor()];

export const ClimateDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { activePreset, customSpec } = useShockLab();

  const [initState, setInitState] = useState<InitialState | null>(null);
  const history3Day = initState?.history3Day ?? PLACEHOLDER_HISTORY;
  const initialCondition = history3Day[2]; // Day 0 state

  const [baselineRollout, setBaselineRollout] = useState<SimulationRollout>([]);
  const [shockRollout, setShockRollout] = useState<SimulationRollout>([]);
  const [selectedChannel, setSelectedChannel] = useState<ChannelId>('tp');
  const [viewMode, setViewMode] = useState<ViewMode>('SHOCK');

  const [apiStatus, setApiStatus] = useState<ApiStatus>({
    serverOnline: false,
    isLive: false,
    fallbackReason: null,
    initialStateLabel: 'loading initial state…',
    initialStateIsSynthetic: false,
    isSimulating: false,
    latencyMs: 0,
    endpoint: 'http://127.0.0.1:8000',
    modelName: 'MonsoonFlow-10',
    lastSimulatedAt: null,
    errorMessage: null,
  });

  // 1. Load the real Day-0 initial condition once
  useEffect(() => {
    let isMounted = true;
    loadInitialState().then((s) => {
      if (!isMounted) return;
      setInitState(s);
      setApiStatus((prev) => ({
        ...prev,
        initialStateLabel: s.label,
        initialStateIsSynthetic: s.isSynthetic,
      }));
    });
    return () => {
      isMounted = false;
    };
  }, []);

  // 2. Periodic health check
  useEffect(() => {
    let isMounted = true;
    const pollHealth = async () => {
      const { isOnline, info } = await checkBackendHealth();
      if (isMounted) {
        setApiStatus((prev) => ({
          ...prev,
          serverOnline: isOnline,
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

  // 3. Baseline + shock, computed together; stale responses are dropped
  const requestIdRef = useRef(0);

  const executeSimulation = useCallback(async () => {
    if (!initState) return;
    const requestId = ++requestIdRef.current;
    setApiStatus((prev) => ({ ...prev, isSimulating: true }));

    const { history3Day: hist, doySincos } = initState;
    const perturbation = resolvePerturbation(activePreset, customSpec, hist[2]);

    const basePromise = runSimulation(hist, doySincos, null);
    const shockPromise: Promise<SimulationResult> = perturbation
      ? runSimulation(hist, doySincos, perturbation)
      : basePromise; // zero shock: identical to baseline by construction
    const [base, shock] = await Promise.all([basePromise, shockPromise]);
    if (requestId !== requestIdRef.current) return;

    let baseRoll = base.rollout;
    let shockRoll = shock.rollout;
    const bothLive = base.isLiveBackend && shock.isLiveBackend;
    let fallbackReason = base.fallbackReason ?? shock.fallbackReason;
    if (!bothLive && (base.isLiveBackend || shock.isLiveBackend)) {
      // Never diff a live rollout against a synthetic one: use synthetic for both
      baseRoll = simulateOfflineRollout(hist[2], null);
      shockRoll = simulateOfflineRollout(hist[2], perturbation);
      fallbackReason = `mixed sources (${fallbackReason}); both shown as synthetic`;
    }

    setBaselineRollout(baseRoll);
    setShockRollout(shockRoll);
    setApiStatus((prev) => ({
      ...prev,
      isSimulating: false,
      isLive: bothLive,
      fallbackReason: bothLive ? null : fallbackReason,
      latencyMs: shock.inferenceTimeMs,
      lastSimulatedAt: new Date(),
      errorMessage: null,
    }));
  }, [initState, activePreset, customSpec]);

  // Re-run whenever the inputs, the shock, or the server's availability change
  const serverOnline = apiStatus.serverOnline;
  useEffect(() => {
    executeSimulation();
  }, [executeSimulation, serverOnline]);

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
