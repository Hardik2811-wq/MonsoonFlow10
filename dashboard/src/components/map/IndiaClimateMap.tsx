/**
 * IndiaClimateMap.tsx - Leaflet 32x36 India domain climate map
 * Bounds: 6.5°N - 37.5°N, 65.5°E - 100.5°E (cell centres), 1.0° cells.
 *
 * Layer stack (bottom -> top), every layer drawn from live tensors or real static data:
 *   Leaflet basemap (own stacking context) -> terrain relief (checkpoints/static.npy orography)
 *   -> field raster (bilinear or exact cells) -> isolines / glowing extreme cores + shock outline
 *   -> wind flow particles or arrows from the displayed (u10, v10) frame.
 * Optional 3D tilt is a CSS perspective transform; hover + brush clicks are inverse-projected.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { GRID_CONFIG, CHANNELS, type ChannelTensor, type Grid2D } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { useShockLab } from '../../context/ShockLabContext';
import { anomalyRgba, channelRgba, isWindChannel } from '../../services/colormaps';
import {
  buildFieldImage,
  buildHillshadeImage,
  contourSegments,
  drawFieldImage,
  gridDelta,
  makeProjector,
  strokeSegments,
  type ContourStyle,
  type Segment,
} from '../../services/fieldRaster';
import { loadStaticFields, type StaticFields } from '../../services/staticFields';
import { resolvePerturbation } from '../../services/perturbation';
import { WindFlow, drawWindArrows, type WindField } from '../../services/windFlow';
import { latLonToGrid, wireToDisplay, displayDeltaString } from '../../services/physicalUnits';
import { ColorbarLegend } from './ColorbarLegend';
import { ChannelSelector } from './ChannelSelector';
import { Terrain3DView } from './Terrain3DView';
import { Box, Crosshair, Grid3x3, Mountain, Spline, Waves, Wind } from 'lucide-react';

interface HoverInfo {
  lat: number;
  lon: number;
  gridI: number;
  gridJ: number;
  baseVal: number;
  shockVal: number;
  delta: number;
}

type WindMode = 'FLOW' | 'ARROWS' | 'OFF';
type Side = 'all' | 'left' | 'right';

interface ContourLayer {
  segs: Segment[];
  style: ContourStyle;
  side: Side;
}

const TILT_DEG = 38;
const TILT_PERSPECTIVE = 1400;
const SMOOTH_PX_PER_CELL = 8;
/** Cells of soft fade drawn outside the domain in SMOOTH mode (CELLS mode keeps the exact hard edge). */
const EDGE_MARGIN_CELLS = 2;
const HALF = GRID_CONFIG.resolutionDeg / 2;
const STAGE_TRANSFORM = `perspective(${TILT_PERSPECTIVE}px) rotateX(${TILT_DEG}deg)`;
/** The glass side panels (w-80 + 12 px margin, see App.tsx) float over the map's left/right edges. */
const SIDE_INSET = 332;
const OVERLAY_GAP = 12;
/** Glass top bar (Header h-14) and bottom bar (Timeline h-[72px]) float over the map too. */
const TOP_BAR_H = 56;
const BOTTOM_BAR_H = 72;
/** Map-chrome rows at the top (channel bar + layer toggles) and bottom (legend) */
const TOP_CHROME = 96;
const DOMAIN_BOUNDS = L.latLngBounds(
  [GRID_CONFIG.latMin - HALF, GRID_CONFIG.lonMin - HALF],
  [GRID_CONFIG.latMax + HALF, GRID_CONFIG.lonMax + HALF]
);
const FIT_PADDING = {
  paddingTopLeft: L.point(SIDE_INSET + OVERLAY_GAP, TOP_BAR_H + TOP_CHROME),
  paddingBottomRight: L.point(SIDE_INSET + OVERLAY_GAP, BOTTOM_BAR_H + OVERLAY_GAP),
};

/** Keep the 32x36 domain filling the free area: fit it, and never allow zooming out past that. */
function fitDomain(map: L.Map): void {
  const z = map.getBoundsZoom(DOMAIN_BOUNDS, false, L.point(2 * (SIDE_INSET + OVERLAY_GAP), TOP_BAR_H + TOP_CHROME + BOTTOM_BAR_H + OVERLAY_GAP));
  map.setMinZoom(z);
  map.fitBounds(DOMAIN_BOUNDS, { ...FIT_PADDING, animate: false });
}

const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Size a canvas to the map in CSS px with a devicePixelRatio backing store; returns a CSS-px context. */
function prepareCanvas(canvas: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function clipSide(ctx: CanvasRenderingContext2D, side: Side, w: number, h: number): void {
  if (side === 'all') return;
  ctx.beginPath();
  if (side === 'left') ctx.rect(0, 0, w / 2, h);
  else ctx.rect(w / 2, 0, w / 2, h);
  ctx.clip();
}

/** Isolines + glowing cores for one displayed grid (physical thresholds, never data-invented). */
function contourLayersFor(grid: Grid2D, channel: string, side: Side): ContourLayer[] {
  if (channel === 'tp') {
    return [
      { segs: contourSegments(grid, 10), style: { color: '#22D3EE', width: 1.4, glow: 7 }, side },
      { segs: contourSegments(grid, 50), style: { color: '#FF3355', width: 2.2, glow: 14 }, side },
    ];
  }
  if (channel === 'msl') {
    const layers: ContourLayer[] = [];
    for (let p = 98400; p <= 102400; p += 200) {
      const deepLow = p <= 100000;
      layers.push({
        segs: contourSegments(grid, p),
        style: deepLow
          ? { color: '#A5B4FC', width: 1.3, glow: 8 }
          : { color: 'rgba(226, 232, 240, 0.32)', width: 0.8, glow: 0 },
        side,
      });
    }
    return layers;
  }
  return [];
}

export const IndiaClimateMap: React.FC = () => {
  const outerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const reliefCanvasRef = useRef<HTMLCanvasElement>(null);
  const fieldCanvasRef = useRef<HTMLCanvasElement>(null);
  const fxCanvasRef = useRef<HTMLCanvasElement>(null);
  const windCanvasRef = useRef<HTMLCanvasElement>(null);
  const drawStaticRef = useRef<() => void>(() => {});
  const windFlowRef = useRef<WindFlow | null>(null);
  const tiltRef = useRef(false);

  const { baselineRollout, shockRollout, initialCondition, selectedChannel, viewMode, apiStatus } =
    useClimateData();
  const { fractionalDay, isPlaying } = useTimeline();
  const { customSpec, updateCustomSpec, isBrushMode, activePreset } = useShockLab();

  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [showGridLines, setShowGridLines] = useState(false);
  const [smooth, setSmooth] = useState(true);
  const [showRelief, setShowRelief] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [windMode, setWindMode] = useState<WindMode>('FLOW');
  const [tilt, setTilt] = useState(false);
  const [staticFields, setStaticFields] = useState<StaticFields | null>(null);

  const chIdx = CHANNELS[selectedChannel].index;
  const isAnomaly = viewMode === 'ANOMALY_DIFF';

  // ---- Live frames with smooth fractional temporal interpolation for 60fps PLAY -------------
  const perturbation = useMemo(
    () => resolvePerturbation(activePreset, customSpec, initialCondition),
    [activePreset, customSpec, initialCondition]
  );

  const shockDay0: ChannelTensor = useMemo(() => {
    if (!perturbation) return initialCondition;
    return initialCondition.map((ch, c) =>
      ch.map((row, i) =>
        row.map((val, j) => {
          const v = val + perturbation[c][i][j];
          // Clamp non-negative physical channels (tp=1, tcwv=5)
          return c === 1 || c === 5 ? Math.max(0, v) : v;
        })
      )
    );
  }, [initialCondition, perturbation]);

  const lerpGrid = useCallback((a: Grid2D, b: Grid2D, t: number): Grid2D => {
    if (t <= 1e-4) return a;
    if (t >= 1 - 1e-4) return b;
    return a.map((row, i) => row.map((v, j) => v + (b[i][j] - v) * t));
  }, []);

  const { baseFrame, shockFrame } = useMemo(() => {
    const d0 = Math.min(10, Math.max(0, Math.floor(fractionalDay)));
    const d1 = Math.min(10, d0 + 1);
    const alpha = fractionalDay - d0;

    const b0 = d0 === 0 ? initialCondition : baselineRollout[d0 - 1] ?? initialCondition;
    const b1 = d1 === 0 ? initialCondition : baselineRollout[d1 - 1] ?? initialCondition;
    const s0 = d0 === 0 ? shockDay0 : shockRollout[d0 - 1] ?? shockDay0;
    const s1 = d1 === 0 ? shockDay0 : shockRollout[d1 - 1] ?? shockDay0;

    if (alpha <= 1e-4 || d0 === d1) {
      return { baseFrame: b0, shockFrame: s0 };
    }
    const baseInterp = b0.map((g, c) =>
      c === chIdx || c === CHANNELS.u10.index || c === CHANNELS.v10.index ? lerpGrid(g, b1[c], alpha) : g
    );
    const shockInterp = s0.map((g, c) =>
      c === chIdx || c === CHANNELS.u10.index || c === CHANNELS.v10.index ? lerpGrid(g, s1[c], alpha) : g
    );
    return { baseFrame: baseInterp, shockFrame: shockInterp };
  }, [fractionalDay, initialCondition, shockDay0, baselineRollout, shockRollout, chIdx, lerpGrid]);

  const currentBaselineGrid = baseFrame[chIdx];
  const currentShockGrid = shockFrame[chIdx];

  const deltaGrid = useMemo(
    () => gridDelta(currentShockGrid, currentBaselineGrid),
    [currentShockGrid, currentBaselineGrid]
  );

  // Raw peak |delta| (for plume-core contours) and the floored scale used by the colour bar
  const rawMaxAbsDelta = useMemo(() => {
    let m = 0;
    for (const row of deltaGrid) for (const d of row) m = Math.max(m, Math.abs(d));
    return m;
  }, [deltaGrid]);
  const maxAbsDelta = Math.max(1.0, rawMaxAbsDelta);

  // ---- Real static terrain -----------------------------------------------------------------
  useEffect(() => {
    let alive = true;
    loadStaticFields().then((s) => {
      if (alive) setStaticFields(s);
    });
    return () => {
      alive = false;
    };
  }, []);

  const reliefImage = useMemo(
    () => (staticFields ? buildHillshadeImage(staticFields.orography, staticFields.landSea, SMOOTH_PX_PER_CELL) : null),
    [staticFields]
  );

  // ---- Field imagery ----------------------------------------------------------------------
  const fieldImages = useMemo(() => {
    const S = smooth ? (isPlaying ? 6 : SMOOTH_PX_PER_CELL) : 1;
    const m = smooth ? EDGE_MARGIN_CELLS : 0;
    const colour = (v: number) => channelRgba(v, selectedChannel);
    if (isAnomaly) {
      return { left: buildFieldImage(deltaGrid, (d) => anomalyRgba(d, maxAbsDelta), S, m), right: null, margin: m };
    }
    if (viewMode === 'BASELINE') return { left: buildFieldImage(currentBaselineGrid, colour, S, m), right: null, margin: m };
    if (viewMode === 'SPLIT') {
      return {
        left: buildFieldImage(currentBaselineGrid, colour, S, m),
        right: buildFieldImage(currentShockGrid, colour, S, m),
        margin: m,
      };
    }
    return { left: buildFieldImage(currentShockGrid, colour, S, m), right: null, margin: m };
  }, [smooth, isPlaying, selectedChannel, isAnomaly, viewMode, deltaGrid, maxAbsDelta, currentBaselineGrid, currentShockGrid]);

  // ---- Isolines and glowing extreme cores -------------------------------------------------
  const contourLayers = useMemo<ContourLayer[]>(() => {
    if (!showContours) return [];
    if (isAnomaly) {
      if (rawMaxAbsDelta <= 1e-6) return [];
      const core = 0.5 * rawMaxAbsDelta; // shock plume core = half of the peak |delta|
      return [
        { segs: contourSegments(deltaGrid, core), style: { color: '#FB923C', width: 1.8, glow: 12 }, side: 'all' },
        { segs: contourSegments(deltaGrid.map((r) => r.map((d) => -d)), core), style: { color: '#38BDF8', width: 1.8, glow: 12 }, side: 'all' },
      ];
    }
    if (viewMode === 'SPLIT') {
      return [
        ...contourLayersFor(currentBaselineGrid, selectedChannel, 'left'),
        ...contourLayersFor(currentShockGrid, selectedChannel, 'right'),
      ];
    }
    return contourLayersFor(viewMode === 'BASELINE' ? currentBaselineGrid : currentShockGrid, selectedChannel, 'all');
  }, [showContours, isAnomaly, rawMaxAbsDelta, deltaGrid, viewMode, currentBaselineGrid, currentShockGrid, selectedChannel]);

  // ---- Wind fields for the flow layer -------------------------------------------------------
  const windFields = useMemo(() => {
    const u = CHANNELS.u10.index;
    const v = CHANNELS.v10.index;
    const base: WindField = { u: baseFrame[u], v: baseFrame[v], speedScale: 1, minSpeed: 0 };
    const shock: WindField = { u: shockFrame[u], v: shockFrame[v], speedScale: 1, minSpeed: 0 };
    const du = gridDelta(shockFrame[u], baseFrame[u]);
    const dv = gridDelta(shockFrame[v], baseFrame[v]);
    let peak = 0;
    for (let i = 0; i < du.length; i++) for (let j = 0; j < du[i].length; j++) peak = Math.max(peak, Math.hypot(du[i][j], dv[i][j]));
    // Δwind is usually small: speed it up (at most x10, direction exact) and only draw where
    // |Δwind| >= 15% of its peak, so noise-level differences are not dressed up as flow.
    const deltaScale = peak > 1e-3 ? Math.min(10, 8 / peak) : 0;
    const delta: WindField = { u: du, v: dv, speedScale: deltaScale, minSpeed: 0.15 * peak };
    return { base, shock, delta, deltaPeak: peak, deltaScale };
  }, [baseFrame, shockFrame]);

  // ---- Leaflet map ------------------------------------------------------------------------
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;
    const map = L.map(mapContainerRef.current, {
      center: [22, 83],
      zoom: 5,
      zoomSnap: 0.25,
      maxZoom: 9,
      // Data exists only inside the model domain: keep it on screen (generous slack for the panels)
      maxBounds: DOMAIN_BOUNDS.pad(0.6),
      maxBoundsViscosity: 1.0,
      zoomControl: false,
      attributionControl: false,
      zoomAnimation: false, // canvases redraw on 'zoom'; an animated tile zoom would drift from them
    });
    // Dark basemap (Esri World Dark Gray, public tier, no API key)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 16, attribution: 'Tiles &copy; Esri' }
    ).addTo(map);

    const redraw = () => drawStaticRef.current();
    map.on('move zoom resize', redraw);
    const refit = () => fitDomain(map);
    map.on('resize', refit);
    fitDomain(map);
    const t = setTimeout(() => {
      map.invalidateSize();
      fitDomain(map);
    }, 150);
    mapInstanceRef.current = map;
    return () => {
      clearTimeout(t);
      map.off('move zoom resize', redraw);
      map.off('resize', refit);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // ---- Static layers: relief, field, contours, outlines -------------------------------------
  const drawStatic = useCallback(() => {
    const map = mapInstanceRef.current;
    const relief = reliefCanvasRef.current;
    const field = fieldCanvasRef.current;
    const fx = fxCanvasRef.current;
    if (!map || !relief || !field || !fx) return;
    const { x: w, y: h } = map.getSize();
    if (w === 0 || h === 0) return;
    const project = makeProjector(map);

    // Terrain relief
    const rctx = prepareCanvas(relief, w, h);
    if (rctx) {
      rctx.clearRect(0, 0, w, h);
      if (showRelief && reliefImage) drawFieldImage(rctx, reliefImage, project, true);
      // Outside the 32x36 domain there is no model data: dim the basemap there. Drawn below the
      // field so the field's soft edge fades over it.
      const [ox0, oy0] = project(GRID_CONFIG.latMax + HALF, GRID_CONFIG.lonMin - HALF);
      const [ox1, oy1] = project(GRID_CONFIG.latMin - HALF, GRID_CONFIG.lonMax + HALF);
      rctx.save();
      rctx.fillStyle = 'rgba(7, 11, 17, 0.5)';
      rctx.beginPath();
      rctx.rect(0, 0, w, h);
      rctx.rect(ox0, oy0, ox1 - ox0, oy1 - oy0);
      rctx.fill('evenodd');
      rctx.restore();
    }

    // Field raster
    const fctx = prepareCanvas(field, w, h);
    if (fctx) {
      fctx.clearRect(0, 0, w, h);
      if (fieldImages.right) {
        fctx.save();
        clipSide(fctx, 'left', w, h);
        drawFieldImage(fctx, fieldImages.left, project, smooth, fieldImages.margin);
        fctx.restore();
        fctx.save();
        clipSide(fctx, 'right', w, h);
        drawFieldImage(fctx, fieldImages.right, project, smooth, fieldImages.margin);
        fctx.restore();
      } else {
        drawFieldImage(fctx, fieldImages.left, project, smooth, fieldImages.margin);
      }
    }

    // Contours, grid, domain and shock outlines
    const xctx = prepareCanvas(fx, w, h);
    if (!xctx) return;
    xctx.clearRect(0, 0, w, h);
    for (const layer of contourLayers) {
      xctx.save();
      clipSide(xctx, layer.side, w, h);
      strokeSegments(xctx, layer.segs, project, layer.style);
      xctx.restore();
    }

    if (showGridLines) {
      xctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
      xctx.lineWidth = 0.5;
      xctx.beginPath();
      for (let k = 0; k <= GRID_CONFIG.nLon; k++) {
        const lon = GRID_CONFIG.lonMin - HALF + k;
        const [xa, ya] = project(GRID_CONFIG.latMin - HALF, lon);
        const [xb, yb] = project(GRID_CONFIG.latMax + HALF, lon);
        xctx.moveTo(xa, ya);
        xctx.lineTo(xb, yb);
      }
      for (let k = 0; k <= GRID_CONFIG.nLat; k++) {
        const lat = GRID_CONFIG.latMin - HALF + k;
        const [xa, ya] = project(lat, GRID_CONFIG.lonMin - HALF);
        const [xb, yb] = project(lat, GRID_CONFIG.lonMax + HALF);
        xctx.moveTo(xa, ya);
        xctx.lineTo(xb, yb);
      }
      xctx.stroke();
    }

    // Valid-data outline: thin crisp line + label, so the soft edge is never mistaken for data
    const [dx0, dy0] = project(GRID_CONFIG.latMax + HALF, GRID_CONFIG.lonMin - HALF);
    const [dx1, dy1] = project(GRID_CONFIG.latMin - HALF, GRID_CONFIG.lonMax + HALF);
    xctx.save();
    xctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    xctx.shadowBlur = 4;
    xctx.strokeStyle = 'rgba(228, 228, 231, 0.5)';
    xctx.lineWidth = 1;
    xctx.strokeRect(Math.round(dx0) + 0.5, Math.round(dy0) + 0.5, Math.round(dx1 - dx0), Math.round(dy1 - dy0));
    xctx.shadowBlur = 0;
    xctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    xctx.textBaseline = 'bottom';
    const label = 'MODEL DOMAIN · 6.5–37.5°N · 65.5–100.5°E';
    const tw = xctx.measureText(label).width;
    xctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    xctx.fillRect(Math.round(dx1) - tw - 10, Math.round(dy1) - 17, tw + 8, 14);
    xctx.fillStyle = 'rgba(212, 212, 216, 0.9)';
    xctx.fillText(label, Math.round(dx1) - tw - 6, Math.round(dy1) - 5);
    xctx.restore();

    // Active shock footprint
    xctx.save();
    xctx.strokeStyle = '#38BDF8';
    xctx.fillStyle = 'rgba(2, 132, 199, 0.10)';
    xctx.lineWidth = 1.5;
    xctx.setLineDash([5, 4]);
    xctx.beginPath();
    let hasOutline = false;
    if (activePreset.box) {
      // Exactly the cells box_mask() selects (cell edges = centres +/- 0.5 deg)
      const [la0, la1, lo0, lo1] = activePreset.box.box;
      const latLo = Math.ceil(la0 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin - HALF;
      const latHi = Math.floor(la1 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin + HALF;
      const lonLo = Math.ceil(lo0 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin - HALF;
      const lonHi = Math.floor(lo1 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin + HALF;
      const [bx0, by0] = project(latHi, lonLo);
      const [bx1, by1] = project(latLo, lonHi);
      xctx.rect(bx0, by0, bx1 - bx0, by1 - by0);
      hasOutline = true;
    } else if (isBrushMode || activePreset.id === 'CUSTOM' || activePreset.spec) {
      const spec = activePreset.id === 'CUSTOM' || isBrushMode ? customSpec : activePreset.spec ?? customSpec;
      const kmPerDeg = GRID_CONFIG.approxKmPerDeg;
      const cosLat = Math.cos((spec.centerLat * Math.PI) / 180);
      for (let k = 0; k <= 72; k++) {
        const th = (k / 72) * 2 * Math.PI;
        const lat = spec.centerLat + (spec.radiusKm / kmPerDeg) * Math.sin(th);
        const lon = spec.centerLon + (spec.radiusKm / (kmPerDeg * cosLat)) * Math.cos(th);
        const [px, py] = project(lat, lon);
        if (k === 0) xctx.moveTo(px, py);
        else xctx.lineTo(px, py);
      }
      hasOutline = true;
    }
    if (hasOutline) {
      xctx.fill();
      xctx.stroke();
    }
    xctx.restore();

    if (viewMode === 'SPLIT') {
      xctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      xctx.lineWidth = 2;
      xctx.setLineDash([6, 4]);
      xctx.beginPath();
      xctx.moveTo(w / 2, 0);
      xctx.lineTo(w / 2, h);
      xctx.stroke();
      xctx.setLineDash([]);
    }
  }, [
    showRelief,
    reliefImage,
    fieldImages,
    smooth,
    contourLayers,
    showGridLines,
    activePreset,
    isBrushMode,
    customSpec,
    viewMode,
  ]);

  useEffect(() => {
    drawStaticRef.current = drawStatic;
    drawStatic();
  }, [drawStatic]);

  // ---- Wind layer (animated particles or static arrows) -------------------------------------
  useEffect(() => {
    const map = mapInstanceRef.current;
    const canvas = windCanvasRef.current;
    if (!map || !canvas) return;

    const colourBySpeed = isWindChannel(selectedChannel) && !isAnomaly;
    const fieldAt = (x: number): WindField => {
      if (isAnomaly) return windFields.delta;
      if (viewMode === 'BASELINE') return windFields.base;
      if (viewMode === 'SPLIT') return x < map.getSize().x / 2 ? windFields.base : windFields.shock;
      return windFields.shock;
    };

    const clear = () => {
      const { x: w, y: h } = map.getSize();
      const ctx = prepareCanvas(canvas, w, h);
      ctx?.clearRect(0, 0, w, h);
      return ctx;
    };

    if (windMode === 'OFF' || (tilt && !!staticFields) || (isAnomaly && windFields.deltaPeak <= 1e-3)) {
      clear();
      return;
    }

    if (windMode === 'ARROWS' || prefersReducedMotion()) {
      const drawArrows = () => {
        const ctx = clear();
        if (ctx) drawWindArrows(ctx, makeProjector(map), fieldAt, colourBySpeed);
      };
      drawArrows();
      map.on('move zoom resize', drawArrows);
      return () => {
        map.off('move zoom resize', drawArrows);
      };
    }

    if (!windFlowRef.current) windFlowRef.current = new WindFlow(1400);
    const flow = windFlowRef.current;
    const onMove = () => {
      clear();
    };
    map.on('move zoom resize', onMove);
    clear();

    let raf = 0;
    const tick = () => {
      const { x: w, y: h } = map.getSize();
      const ctx = prepareCanvas(canvas, w, h);
      if (ctx) {
        // Fade previous trails
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = 'destination-in';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.92)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        flow.step(ctx, makeProjector(map), fieldAt, colourBySpeed);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      map.off('move zoom resize', onMove);
    };
  }, [windMode, windFields, viewMode, isAnomaly, selectedChannel, tilt, staticFields]);

  // ---- 3D tilt: lock pan/zoom (Leaflet's own hit-testing assumes an untransformed map) ----
  useEffect(() => {
    tiltRef.current = tilt;
    const map = mapInstanceRef.current;
    if (!map) return;
    const handlers = [map.dragging, map.scrollWheelZoom, map.doubleClickZoom, map.boxZoom, map.keyboard];
    handlers.forEach((hd) => (tilt ? hd.disable() : hd.enable()));
  }, [tilt]);

  /** Screen point -> Leaflet container point, inverting the CSS perspective tilt when active. */
  const toContainerPoint = useCallback(
    (clientX: number, clientY: number): L.Point | null => {
      const outer = outerRef.current;
      if (!outer) return null;
      const r = outer.getBoundingClientRect();
      const x = clientX - r.left;
      const y = clientY - r.top;
      if (!tilt) return L.point(x, y);
      const th = (TILT_DEG * Math.PI) / 180;
      const P = TILT_PERSPECTIVE;
      const xs = x - r.width / 2;
      const ys = y - r.height / 2;
      const denom = P * Math.cos(th) + ys * Math.sin(th);
      if (denom <= 0) return null; // above the horizon
      const vy = (ys * P) / denom;
      const vx = (xs * (P - vy * Math.sin(th))) / P;
      return L.point(vx + r.width / 2, vy + r.height / 2);
    },
    [tilt]
  );

  // Brush painting: Leaflet click when flat; our own inverse-projected click when tilted
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const handleMapClick = (e: L.LeafletMouseEvent) => {
      if (!isBrushMode || tiltRef.current) return;
      paintAt(e.latlng.lat, e.latlng.lng);
    };
    const paintAt = (lat: number, lng: number) => {
      if (lat >= GRID_CONFIG.latMin && lat <= GRID_CONFIG.latMax && lng >= GRID_CONFIG.lonMin && lng <= GRID_CONFIG.lonMax) {
        updateCustomSpec({ centerLat: Math.round(lat * 10) / 10, centerLon: Math.round(lng * 10) / 10 });
      }
    };
    map.on('click', handleMapClick);
    return () => {
      map.off('click', handleMapClick);
    };
  }, [isBrushMode, updateCustomSpec]);

  const handleTiltedClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const map = mapInstanceRef.current;
    if (!tilt || !isBrushMode || !map || !stageRef.current?.contains(e.target as Node)) return;
    const pt = toContainerPoint(e.clientX, e.clientY);
    if (!pt) return;
    const ll = map.containerPointToLatLng(pt);
    if (ll.lat >= GRID_CONFIG.latMin && ll.lat <= GRID_CONFIG.latMax && ll.lng >= GRID_CONFIG.lonMin && ll.lng <= GRID_CONFIG.lonMax) {
      updateCustomSpec({ centerLat: Math.round(ll.lat * 10) / 10, centerLon: Math.round(ll.lng * 10) / 10 });
    }
  };

  // ---- Hover inspector & 3D callbacks --------------------------------------------------------
  const setHoverFromLatLon = useCallback(
    (lat: number | null, lon: number | null) => {
      if (lat === null || lon === null) {
        setHoverInfo(null);
        return;
      }
      const grid = latLonToGrid(lat, lon);
      if (!grid) {
        setHoverInfo(null);
        return;
      }
      const baseVal = currentBaselineGrid[grid.i][grid.j];
      const shockVal = currentShockGrid[grid.i][grid.j];
      setHoverInfo({
        lat: Math.round(lat * 10) / 10,
        lon: Math.round(lon * 10) / 10,
        gridI: grid.i,
        gridJ: grid.j,
        baseVal,
        shockVal,
        delta: shockVal - baseVal,
      });
    },
    [currentBaselineGrid, currentShockGrid]
  );

  const handleBrushLatLon = useCallback(
    (lat: number, lon: number) => {
      if (!isBrushMode) return;
      if (lat >= GRID_CONFIG.latMin && lat <= GRID_CONFIG.latMax && lon >= GRID_CONFIG.lonMin && lon <= GRID_CONFIG.lonMax) {
        updateCustomSpec({ centerLat: Math.round(lat * 10) / 10, centerLon: Math.round(lon * 10) / 10 });
      }
    },
    [isBrushMode, updateCustomSpec]
  );

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (tilt && staticFields) return; // handled by Terrain3DView raycaster
    const map = mapInstanceRef.current;
    if (!map) return;
    const pt = toContainerPoint(e.clientX, e.clientY);
    const latLng = pt ? map.containerPointToLatLng(pt) : null;
    if (latLng) setHoverFromLatLon(latLng.lat, latLng.lng);
    else setHoverInfo(null);
  };

  const toggleBtn = (active: boolean) =>
    `px-2 py-1 rounded transition-colors flex items-center gap-1 ${
      active ? 'bg-white/10 text-white font-semibold' : 'hover:text-zinc-200'
    }`;

  const nextWindMode: Record<WindMode, WindMode> = { FLOW: 'ARROWS', ARROWS: 'OFF', OFF: 'FLOW' };

  return (
    <div
      ref={outerRef}
      className="relative w-full h-full bg-[#050505] overflow-hidden select-none"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverInfo(null)}
      onClick={handleTiltedClick}
    >
      {/* 2D Leaflet stage (untouched when 3D is off) */}
      <div
        ref={stageRef}
        className="absolute inset-0 transition-transform duration-500 ease-out"
        style={{
          transform: tilt && !staticFields ? STAGE_TRANSFORM : 'none',
          transformOrigin: '50% 50%',
          boxShadow: tilt && !staticFields ? '0 30px 80px rgba(0, 0, 0, 0.7)' : 'none',
        }}
      >
        {/* Leaflet in its own stacking context so its z-index 400 panes stay below the overlays */}
        <div ref={mapContainerRef} className="absolute inset-0 isolate z-0" />
        <canvas ref={reliefCanvasRef} className="absolute inset-0 pointer-events-none z-[5]" />
        <canvas
          ref={fieldCanvasRef}
          className="absolute inset-0 pointer-events-none z-10"
          style={{ filter: 'saturate(0.85) drop-shadow(0 2px 3px rgba(0, 0, 0, 0.55))' }}
        />
        <canvas ref={fxCanvasRef} className="absolute inset-0 pointer-events-none z-[11]" />
        <canvas ref={windCanvasRef} className="absolute inset-0 pointer-events-none z-[12]" />
      </div>

      {/* True 3D Terrain Heightfield (kept mounted so WebGL & Esri tiles stay cached; crossfades smoothly) */}
      {staticFields && (
        <Terrain3DView
          active={tilt}
          staticFields={staticFields}
          fieldImages={fieldImages}
          smooth={smooth}
          contourLayers={contourLayers}
          showGridLines={showGridLines}
          viewMode={viewMode}
          selectedChannel={selectedChannel}
          windMode={windMode}
          windFields={windFields}
          activePreset={activePreset}
          customSpec={customSpec}
          isBrushMode={isBrushMode}
          onHoverLatLon={setHoverFromLatLon}
          onBrushLatLon={handleBrushLatLon}
        />
      )}

      {/* Atmospheric depth haze toward the horizon only for fallback CSS tilt */}
      {tilt && !staticFields && (
        <div className="absolute inset-x-0 top-0 h-2/5 z-[15] pointer-events-none bg-gradient-to-b from-[#050505] via-[#050505]/60 to-transparent" />
      )}

      {viewMode === 'SPLIT' && (
        <div className="absolute z-20 flex justify-between pointer-events-none text-xs font-mono" style={{ top: TOP_BAR_H + 96, left: SIDE_INSET + 3 * OVERLAY_GAP, right: SIDE_INSET + 3 * OVERLAY_GAP }}>
          <div className="bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] px-3 py-1 rounded text-zinc-300 shadow">
            ← BASELINE (P = 0)
          </div>
          <div className="bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/25 px-3 py-1 rounded text-white shadow">
            SHOCK MULTIVERSE →
          </div>
        </div>
      )}

      {/* Top-left: channel + layer toggles */}
      <div className="absolute z-20 flex flex-col items-start gap-2" style={{ top: TOP_BAR_H + OVERLAY_GAP, left: SIDE_INSET + OVERLAY_GAP }}>
        <ChannelSelector />
        <div className="flex items-center bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] rounded p-1 text-xs font-mono text-zinc-400 gap-0.5">
          <button onClick={() => setSmooth((p) => !p)} className={toggleBtn(smooth)} title="Bilinear interpolation between cell centres (off: exact 1° cells)">
            <Waves className="w-3.5 h-3.5" />
            <span>{smooth ? 'SMOOTH' : 'CELLS'}</span>
          </button>
          <button onClick={() => setShowRelief((p) => !p)} className={toggleBtn(showRelief)} disabled={!staticFields} title={staticFields ? 'Terrain relief from orography (static.npy)' : 'static_fields.json not available'}>
            <Mountain className="w-3.5 h-3.5" />
            <span>RELIEF</span>
          </button>
          <button onClick={() => setShowContours((p) => !p)} className={toggleBtn(showContours)} title="Isolines and glowing extreme cores">
            <Spline className="w-3.5 h-3.5" />
            <span>CONTOURS</span>
          </button>
          <button onClick={() => setWindMode((m) => nextWindMode[m])} className={toggleBtn(windMode !== 'OFF')} title="Wind from the displayed u10/v10 frame: flow particles → arrows → off">
            <Wind className="w-3.5 h-3.5" />
            <span>WIND {windMode}</span>
          </button>
          <button onClick={() => setShowGridLines((p) => !p)} className={toggleBtn(showGridLines)} title="Toggle 1.0° grid lines">
            <Grid3x3 className="w-3.5 h-3.5" />
            <span>GRID</span>
          </button>
          <button onClick={() => setTilt((p) => !p)} className={toggleBtn(tilt)} title="True 3D terrain elevation (orography_m) — drag to orbit, scroll to zoom">
            <Box className="w-3.5 h-3.5" />
            <span>3D</span>
          </button>
        </div>
        {tilt && (
          <div className="px-2 py-0.5 rounded bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] text-[10px] font-mono text-zinc-400">
            3D terrain (orography_m) · drag to orbit · scroll to zoom · inspector &amp; brush active
          </div>
        )}
        {isAnomaly && windMode !== 'OFF' && windFields.deltaPeak > 1e-3 && (
          <div className="px-2 py-0.5 rounded bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] text-[10px] font-mono text-zinc-500">
            wind = Δ(u10, v10) · peak {windFields.deltaPeak.toFixed(2)} m/s · drawn ×{windFields.deltaScale.toFixed(1)} where ≥ 15% of peak
          </div>
        )}
      </div>

      {/* Top-right: domain + Day-0 provenance */}
      <div className="absolute z-20 flex flex-col items-end gap-2" style={{ top: TOP_BAR_H + OVERLAY_GAP, right: SIDE_INSET + OVERLAY_GAP }}>
        <div className="px-2.5 py-1 rounded bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] text-xs font-mono text-zinc-300">
          DOMAIN: <span className="text-white">32x36 (1.0° res)</span>
        </div>
        <div
          className={`px-2.5 py-1 rounded bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border text-xs font-mono ${
            apiStatus.initialStateIsSynthetic ? 'border-rose-800 text-rose-300' : 'border-white/[0.12] text-zinc-300'
          }`}
        >
          DAY 0: <span className={apiStatus.initialStateIsSynthetic ? '' : 'text-white'}>{apiStatus.initialStateLabel}</span>
        </div>
      </div>

      <div className="absolute z-20" style={{ bottom: BOTTOM_BAR_H + OVERLAY_GAP, left: SIDE_INSET + OVERLAY_GAP }}>
        <ColorbarLegend isAnomaly={isAnomaly} maxAbsDelta={maxAbsDelta} contoursOn={showContours} />
      </div>

      {hoverInfo && (
        <div style={{ right: SIDE_INSET + OVERLAY_GAP, bottom: BOTTOM_BAR_H + OVERLAY_GAP }} className="absolute z-20 bg-black/30 backdrop-blur-2xl backdrop-saturate-150 border border-white/[0.12] rounded p-3 text-xs font-mono shadow-2xl tabular-nums">
          <div className="flex items-center gap-2 mb-1.5 text-zinc-400 border-b border-white/[0.12] pb-1">
            <Crosshair className="w-3.5 h-3.5 text-zinc-300" />
            <span>
              {hoverInfo.lat.toFixed(1)}°N, {hoverInfo.lon.toFixed(1)}°E
            </span>
            <span className="text-zinc-600">|</span>
            <span className="text-zinc-500">
              [{hoverInfo.gridI}, {hoverInfo.gridJ}]
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <span className="text-zinc-500 block text-[10px]">BASELINE</span>
              <span className="text-zinc-200 font-medium">
                {wireToDisplay(hoverInfo.baseVal, selectedChannel).str} {wireToDisplay(hoverInfo.baseVal, selectedChannel).unit}
              </span>
            </div>
            <div>
              <span className="text-zinc-500 block text-[10px]">SHOCK</span>
              <span className="text-white font-medium">
                {wireToDisplay(hoverInfo.shockVal, selectedChannel).str} {wireToDisplay(hoverInfo.shockVal, selectedChannel).unit}
              </span>
            </div>
            <div>
              <span className="text-zinc-500 block text-[10px]">ANOMALY Δ</span>
              <span
                className={`font-semibold ${
                  hoverInfo.delta > 0 ? 'text-orange-400' : hoverInfo.delta < 0 ? 'text-sky-400' : 'text-zinc-400'
                }`}
              >
                {displayDeltaString(hoverInfo.delta, selectedChannel)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

