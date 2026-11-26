/**
 * colormaps.ts - Meteorological palettes and divergent anomaly scales
 */

import type { ChannelId } from '../types/climate';

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export function interpolateRgb(c1: RgbColor, c2: RgbColor, t: number): RgbColor {
  const clamped = Math.max(0, Math.min(1, t));
  return {
    r: Math.round(c1.r + (c2.r - c1.r) * clamped),
    g: Math.round(c1.g + (c2.g - c1.g) * clamped),
    b: Math.round(c1.b + (c2.b - c1.b) * clamped),
  };
}

export function rgbToCss(c: RgbColor, alpha = 1): string {
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
}

// Temperature: Blue -> Cyan -> Yellow -> Orange -> Deep Red
const TEMP_STOPS: [number, RgbColor][] = [
  [280, { r: 30, g: 58, b: 138 }],   // Deep Navy / 7°C
  [290, { r: 59, g: 130, b: 246 }],  // Blue / 17°C
  [298, { r: 34, g: 197, b: 94 }],   // Green / 25°C
  [305, { r: 234, g: 179, b: 8 }],   // Yellow / 32°C
  [312, { r: 249, g: 115, b: 22 }],  // Orange / 39°C
  [318, { r: 220, g: 38, b: 38 }],   // Red / 45°C
  [323, { r: 127, g: 29, b: 29 }],   // Dark Maroon / 50°C
];

// Precipitation: White/transparent -> Light Cyan -> Sky Blue -> Navy -> Violet
const PRECIP_STOPS: [number, RgbColor][] = [
  [0, { r: 15, g: 23, b: 42 }],      // 0 mm (Dark neutral slate)
  [2, { r: 56, g: 189, b: 248 }],    // 2 mm (Light cyan)
  [10, { r: 2, g: 132, b: 199 }],    // 10 mm (Moderate blue)
  [25, { r: 30, g: 64, b: 175 }],    // 25 mm (Deep navy)
  [50, { r: 147, g: 51, b: 234 }],   // 50 mm (Extreme violet)
  [80, { r: 244, g: 63, b: 94 }],    // 80+ mm (Severe crimson/magenta)
];

// Pressure (MSL): Deep Indigo -> Blue -> Teal -> Gold
const MSL_STOPS: [number, RgbColor][] = [
  [99200, { r: 88, g: 28, b: 135 }],  // Deep Low
  [99800, { r: 2, g: 132, b: 199 }],
  [100600, { r: 16, g: 185, b: 129 }],
  [101400, { r: 234, g: 179, b: 8 }], // High
];

// TCWV: Dry Pale Sand -> Mint -> Cerulean
const TCWV_STOPS: [number, RgbColor][] = [
  [15, { r: 120, g: 113, b: 108 }], // Dry
  [30, { r: 20, g: 184, b: 166 }],  // Moderate
  [50, { r: 2, g: 132, b: 199 }],   // Heavy
  [70, { r: 79, g: 70, b: 229 }],   // Super-saturated
];

function evaluateMultiStop(value: number, stops: [number, RgbColor][]): RgbColor {
  if (value <= stops[0][0]) return stops[0][1];
  if (value >= stops[stops.length - 1][0]) return stops[stops.length - 1][1];

  for (let i = 0; i < stops.length - 1; i++) {
    const [v0, c0] = stops[i];
    const [v1, c1] = stops[i + 1];
    if (value >= v0 && value <= v1) {
      const t = (value - v0) / (v1 - v0);
      return interpolateRgb(c0, c1, t);
    }
  }
  return stops[stops.length - 1][1];
}

export function getColorForChannel(value: number, channel: ChannelId): RgbColor {
  switch (channel) {
    case 't2m':
      return evaluateMultiStop(value, TEMP_STOPS);
    case 'tp':
      return evaluateMultiStop(value, PRECIP_STOPS);
    case 'msl':
      return evaluateMultiStop(value, MSL_STOPS);
    case 'tcwv':
      return evaluateMultiStop(value, TCWV_STOPS);
    case 'u10':
    case 'v10': {
      // Magnitude scale for wind velocity components
      const mag = Math.abs(value);
      const t = Math.min(1, mag / 20.0);
      return interpolateRgb({ r: 30, g: 41, b: 59 }, { r: 56, g: 189, b: 248 }, t);
    }
    default:
      return { r: 148, g: 163, b: 184 };
  }
}

/**
 * Divergent colormap for Anomaly Diff: Delta X_t = X_t(shock) - X_t(baseline)
 * Blue (Negative anomaly) <-> Neutral Charcoal (Zero) <-> Amber/Red (Positive anomaly)
 */
export function getAnomalyDivergentColor(delta: number, maxAbsDelta: number): RgbColor {
  const norm = maxAbsDelta > 0 ? Math.max(-1, Math.min(1, delta / maxAbsDelta)) : 0;
  const zeroColor: RgbColor = { r: 27, g: 38, b: 59 }; // Neutral slate charcoal

  if (norm < 0) {
    // Negative anomaly -> Cold Cyan/Blue
    const negCold: RgbColor = { r: 56, g: 189, b: 248 };
    return interpolateRgb(zeroColor, negCold, -norm);
  } else {
    // Positive anomaly -> Warm Red/Amber
    const posWarm: RgbColor = { r: 239, g: 68, b: 68 };
    return interpolateRgb(zeroColor, posWarm, norm);
  }
}
