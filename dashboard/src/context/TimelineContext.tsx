/**
 * TimelineContext.tsx - Day 0 to 10 forecast timeline scrubber engine
 * Supports smooth 60fps fractional playback via requestAnimationFrame while keeping integer
 * currentDay (0..10) for discrete keyframe readouts.
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';

interface TimelineContextType {
  currentDay: number; // rounded integer 0..10 for badges & keyframe pills
  fractionalDay: number; // continuous 0.0..10.0 for smooth 60fps playback interpolation
  isPlaying: boolean;
  playbackSpeed: number; // 0.5, 1, 2
  setCurrentDay: (day: number) => void;
  setFractionalDay: (day: number) => void;
  togglePlay: () => void;
  setPlaybackSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  resetToDay0: () => void;
}

const TimelineContext = createContext<TimelineContextType | undefined>(undefined);

/** Days per second at 1x playback speed */
const BASE_DAYS_PER_SEC = 0.85;

export const TimelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [fractionalDay, setFractionalDayState] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const speedRef = useRef(playbackSpeed);

  useEffect(() => {
    speedRef.current = playbackSpeed;
  }, [playbackSpeed]);

  const currentDay = Math.min(10, Math.max(0, Math.round(fractionalDay)));

  const setFractionalDay = useCallback((day: number) => {
    setFractionalDayState(Math.min(10, Math.max(0, day)));
  }, []);

  const setCurrentDay = useCallback((day: number) => {
    setFractionalDayState(Math.min(10, Math.max(0, Math.round(day))));
  }, []);

  const stepForward = useCallback(() => {
    setFractionalDayState((prev) => {
      const d = Math.round(prev);
      return d >= 10 ? 0 : d + 1;
    });
  }, []);

  const stepBackward = useCallback(() => {
    setFractionalDayState((prev) => {
      const d = Math.round(prev);
      return d <= 0 ? 10 : d - 1;
    });
  }, []);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const resetToDay0 = useCallback(() => {
    setIsPlaying(false);
    setFractionalDayState(0);
  }, []);

  useEffect(() => {
    if (!isPlaying) return;

    let raf = 0;
    let lastMs = performance.now();

    const tick = (now: number) => {
      const dtSec = Math.min(0.1, (now - lastMs) / 1000);
      lastMs = now;
      setFractionalDayState((prev) => {
        const next = prev + dtSec * BASE_DAYS_PER_SEC * speedRef.current;
        return next > 10 ? 0 : next;
      });
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying]);

  return (
    <TimelineContext.Provider
      value={{
        currentDay,
        fractionalDay,
        isPlaying,
        playbackSpeed,
        setCurrentDay,
        setFractionalDay,
        togglePlay,
        setPlaybackSpeed,
        stepForward,
        stepBackward,
        resetToDay0,
      }}
    >
      {children}
    </TimelineContext.Provider>
  );
};

export function useTimeline(): TimelineContextType {
  const ctx = useContext(TimelineContext);
  if (!ctx) throw new Error('useTimeline must be used within TimelineProvider');
  return ctx;
}
