/**
 * Terrain3DView.tsx - True 3D terrain heightfield using real orography_m & land_sea_mask
 *
 * - Upsamples 32x36 orography_m to 96x108 with bilinear interpolation and extrudes real Z height.
 * - Composites Esri World Dark Gray basemap tiles (with CORS) + land_sea_mask=0.5 coastlines
 *   directly into the 3D surface texture underneath the translucent weather field.
 * - Renders live WindFlow particles / arrows and crisp contours onto the 3D surface.
 * - Supports damped drag-to-orbit, scroll zoom, and raycasted hover/brush interaction.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { GRID_CONFIG, type ChannelId } from '../../types/climate';
import type { ViewMode } from '../../context/ClimateDataContext';
import type { StaticFields } from '../../services/staticFields';
import type { GaussianShockSpec, ShockPreset } from '../../types/shocks';
import { isWindChannel } from '../../services/colormaps';
import {
  contourSegments,
  sampleBilinear,
  type ContourStyle,
  type Projector,
  type Segment,
} from '../../services/fieldRaster';
import { WindFlow, drawWindArrows, type WindField } from '../../services/windFlow';

export interface TerrainContourLayer {
  segs: Segment[];
  style: ContourStyle;
  side: 'all' | 'left' | 'right';
}

export type TerrainWindMode = 'FLOW' | 'ARROWS' | 'OFF';

interface Terrain3DViewProps {
  active: boolean;
  staticFields: StaticFields;
  reliefImage: HTMLCanvasElement | null;
  showRelief: boolean;
  fieldImages: {
    left: HTMLCanvasElement;
    right: HTMLCanvasElement | null;
    margin: number;
  };
  smooth: boolean;
  contourLayers: TerrainContourLayer[];
  showGridLines: boolean;
  viewMode: ViewMode;
  selectedChannel: ChannelId;
  windMode: TerrainWindMode;
  windFields: {
    base: WindField;
    shock: WindField;
    delta: WindField;
    deltaPeak: number;
  };
  activePreset: ShockPreset;
  customSpec: GaussianShockSpec;
  isBrushMode: boolean;
  onHoverLatLon: (lat: number | null, lon: number | null) => void;
  onBrushLatLon: (lat: number, lon: number) => void;
}

const NLAT = GRID_CONFIG.nLat; // 32
const NLON = GRID_CONFIG.nLon; // 36
const HALF = GRID_CONFIG.resolutionDeg / 2; // 0.5
const LON_MIN = GRID_CONFIG.lonMin - HALF; // 65.0
const LON_MAX = GRID_CONFIG.lonMax + HALF; // 101.0
const LAT_MIN = GRID_CONFIG.latMin - HALF; // 6.0
const LAT_MAX = GRID_CONFIG.latMax + HALF; // 38.0

/** 5x bicubic upsampling -> 160x180 sculptural terrain mesh */
const SEG_X = 180;
const SEG_Z = 160;

/** World dimensions preserving cos(22°) aspect ratio of the 32°x36° India domain */
const WORLD_W = NLON * Math.cos((22 * Math.PI) / 180); // ~33.38
const WORLD_D = NLAT; // 32
const MAX_OROGRAPHY_M = 5284;
const MAX_HEIGHT = 6.2;

/** Draped texture resolution (24 px per 1° cell = 864x768 for crisp labels, coastlines & wind) */
const TEX_S = 24;
const TEX_W = NLON * TEX_S;
const TEX_H = NLAT * TEX_S;

const ESRI_TILE_URL =
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile';
const ESRI_ZOOM = 5;

const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const mercYNorm = (lat: number): number => {
  const rad = (lat * Math.PI) / 180;
  return (1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2;
};

const lonXNorm = (lon: number): number => (lon + 180) / 360;

const toTexProjector: Projector = (lat, lon) => [
  (lon - LON_MIN) * TEX_S,
  (LAT_MAX - lat) * TEX_S,
];

function cubic1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  const a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
  const a2 = -0.5 * p0 + 0.5 * p2;
  const a3 = p1;
  return ((a0 * t + a1) * t + a2) * t + a3;
}

/** Smooth bicubic Catmull-Rom sample for sculptural ridge geometry on 160x180 mesh. */
function sampleBicubic(grid: number[][], fi: number, fj: number): number {
  const i = Math.max(0, Math.min(NLAT - 1, fi));
  const j = Math.max(0, Math.min(NLON - 1, fj));
  const i1 = Math.floor(i);
  const j1 = Math.floor(j);
  const ti = i - i1;
  const tj = j - j1;
  const clampI = (k: number) => Math.max(0, Math.min(NLAT - 1, k));
  const clampJ = (k: number) => Math.max(0, Math.min(NLON - 1, k));
  const rowVals = [0, 0, 0, 0];
  for (let m = -1; m <= 2; m++) {
    const r = grid[clampI(i1 + m)];
    rowVals[m + 1] = cubic1D(
      r[clampJ(j1 - 1)],
      r[clampJ(j1)],
      r[clampJ(j1 + 1)],
      r[clampJ(j1 + 2)],
      tj
    );
  }
  return Math.max(0, cubic1D(rowVals[0], rowVals[1], rowVals[2], rowVals[3], ti));
}

function worldToLatLon(x: number, z: number): { lat: number; lon: number } {
  const u = x / WORLD_W + 0.5;
  const v = z / WORLD_D + 0.5;
  const lon = LON_MIN + u * NLON;
  const lat = LAT_MAX - v * NLAT;
  return { lat, lon };
}

/**
 * Fetch Esri World Dark Gray tiles at zoom 5 covering [6..38N, 65..101E] with CORS,
 * and reproject row-by-row from Web Mercator onto our linear (lon, lat) texture canvas.
 */
async function loadEsriDomainBasemap(): Promise<HTMLCanvasElement | null> {
  const n = 1 << ESRI_ZOOM; // 32
  const x0 = Math.floor(lonXNorm(LON_MIN) * n);
  const x1 = Math.floor(lonXNorm(LON_MAX) * n);
  const y0 = Math.floor(mercYNorm(LAT_MAX) * n);
  const y1 = Math.floor(mercYNorm(LAT_MIN) * n);

  const tilesW = (x1 - x0 + 1) * 256;
  const tilesH = (y1 - y0 + 1) * 256;
  const stitched = document.createElement('canvas');
  stitched.width = tilesW;
  stitched.height = tilesH;
  const sctx = stitched.getContext('2d');
  if (!sctx) return null;

  const jobs: Promise<boolean>[] = [];
  for (let ty = y0; ty <= y1; ty++) {
    for (let tx = x0; tx <= x1; tx++) {
      jobs.push(
        new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            sctx.drawImage(img, (tx - x0) * 256, (ty - y0) * 256);
            resolve(true);
          };
          img.onerror = () => resolve(false);
          img.src = `${ESRI_TILE_URL}/${ESRI_ZOOM}/${ty}/${tx}`;
        })
      );
    }
  }

  const results = await Promise.all(jobs);
  if (!results.some(Boolean)) return null;

  const out = document.createElement('canvas');
  out.width = TEX_W;
  out.height = TEX_H;
  const octx = out.getContext('2d');
  if (!octx) return null;

  const sxLeft = (lonXNorm(LON_MIN) * n - x0) * 256;
  const sxRight = (lonXNorm(LON_MAX) * n - x0) * 256;
  const sxW = sxRight - sxLeft;

  octx.save();
  octx.filter = 'brightness(1.30) contrast(1.20)';
  octx.imageSmoothingEnabled = true;
  // Reproject Web Mercator Y to linear latitude in 2px strips
  for (let py = 0; py < TEX_H; py += 2) {
    const latTop = LAT_MAX - (py / TEX_H) * NLAT;
    const latBot = LAT_MAX - ((py + 2) / TEX_H) * NLAT;
    const syTop = (mercYNorm(latTop) * n - y0) * 256;
    const syBot = (mercYNorm(latBot) * n - y0) * 256;
    const sh = Math.max(0.5, syBot - syTop);
    octx.drawImage(stitched, sxLeft, syTop, sxW, sh, 0, py, TEX_W, 2);
  }
  octx.restore();
  return out;
}

function drawStaticDrapedCanvas(
  canvas: HTMLCanvasElement,
  baseCanvas: HTMLCanvasElement,
  reliefImage: HTMLCanvasElement | null,
  showRelief: boolean,
  fieldImages: Terrain3DViewProps['fieldImages'],
  smooth: boolean,
  contourLayers: TerrainContourLayer[],
  showGridLines: boolean,
  viewMode: ViewMode,
  activePreset: ShockPreset,
  customSpec: GaussianShockSpec,
  isBrushMode: boolean
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, TEX_W, TEX_H);
  // 1. Esri basemap + land/ocean contrast + crisp coastline contour
  ctx.drawImage(baseCanvas, 0, 0);

  // 1b. Orographic hillshade relief (matching 2D RELIEF toggle)
  if (showRelief && reliefImage) {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(reliefImage, 0, 0, TEX_W, TEX_H);
    ctx.restore();
  }

  // 2. Draped weather field (cropped to inner 32x36 domain)
  const margin = fieldImages.margin;
  const imgS = fieldImages.left.height / (NLAT + 2 * margin);
  const sx = margin * imgS;
  const sy = margin * imgS;
  const sw = NLON * imgS;
  const sh = NLAT * imgS;

  ctx.save();
  ctx.imageSmoothingEnabled = smooth;
  if (fieldImages.right) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, TEX_W / 2, TEX_H);
    ctx.clip();
    ctx.drawImage(fieldImages.left, sx, sy, sw, sh, 0, 0, TEX_W, TEX_H);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(TEX_W / 2, 0, TEX_W / 2, TEX_H);
    ctx.clip();
    ctx.drawImage(fieldImages.right, sx, sy, sw, sh, 0, 0, TEX_W, TEX_H);
    ctx.restore();
  } else {
    ctx.drawImage(fieldImages.left, sx, sy, sw, sh, 0, 0, TEX_W, TEX_H);
  }
  ctx.restore();

  const toTex = (fi: number, fj: number): [number, number] => [
    (fj + 0.5) * TEX_S,
    (NLAT - 0.5 - fi) * TEX_S,
  ];

  // 3. Marching-squares contours + glowing extreme cores
  for (const layer of contourLayers) {
    if (layer.segs.length === 0) continue;
    ctx.save();
    if (layer.side !== 'all') {
      ctx.beginPath();
      if (layer.side === 'left') ctx.rect(0, 0, TEX_W / 2, TEX_H);
      else ctx.rect(TEX_W / 2, 0, TEX_W / 2, TEX_H);
      ctx.clip();
    }
    ctx.beginPath();
    for (const [i1, j1, i2, j2] of layer.segs) {
      const [x1, y1] = toTex(i1, j1);
      const [x2, y2] = toTex(i2, j2);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.lineCap = 'round';
    ctx.setLineDash(layer.style.dash ?? []);
    ctx.strokeStyle = layer.style.color;
    ctx.lineWidth = layer.style.width * 1.2;
    if (layer.style.glow > 0) {
      ctx.shadowColor = layer.style.color;
      ctx.shadowBlur = layer.style.glow;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 0.92;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
      ctx.lineWidth = Math.max(0.8, layer.style.width * 0.55);
    }
    ctx.stroke();
    ctx.restore();
  }

  // 4. Grid lines
  if (showGridLines) {
    ctx.save();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.24)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let k = 0; k <= NLON; k++) {
      const x = k * TEX_S;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, TEX_H);
    }
    for (let k = 0; k <= NLAT; k++) {
      const y = k * TEX_S;
      ctx.moveTo(0, y);
      ctx.lineTo(TEX_W, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // 5. Active shock footprint draped onto terrain (clipped to right/shock half in SPLIT view)
  ctx.save();
  if (viewMode === 'SPLIT') {
    ctx.beginPath();
    ctx.rect(TEX_W / 2, 0, TEX_W / 2, TEX_H);
    ctx.clip();
  }
  ctx.strokeStyle = '#38BDF8';
  ctx.fillStyle = 'rgba(2, 132, 199, 0.14)';
  ctx.lineWidth = 2.2;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  let hasOutline = false;
  if (activePreset.box) {
    const [la0, la1, lo0, lo1] = activePreset.box.box;
    const latLo = Math.ceil(la0 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin - HALF;
    const latHi = Math.floor(la1 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin + HALF;
    const lonLo = Math.ceil(lo0 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin - HALF;
    const lonHi = Math.floor(lo1 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin + HALF;
    const [bx0, by0] = toTexProjector(latHi, lonLo);
    const [bx1, by1] = toTexProjector(latLo, lonHi);
    ctx.rect(bx0, by0, bx1 - bx0, by1 - by0);
    hasOutline = true;
  } else if (isBrushMode || activePreset.id === 'CUSTOM' || activePreset.spec) {
    const spec = activePreset.id === 'CUSTOM' || isBrushMode ? customSpec : activePreset.spec ?? customSpec;
    const kmPerDeg = GRID_CONFIG.approxKmPerDeg;
    const cosLat = Math.cos((spec.centerLat * Math.PI) / 180);
    for (let k = 0; k <= 72; k++) {
      const th = (k / 72) * 2 * Math.PI;
      const lat = spec.centerLat + (spec.radiusKm / kmPerDeg) * Math.sin(th);
      const lon = spec.centerLon + (spec.radiusKm / (kmPerDeg * cosLat)) * Math.cos(th);
      const [px, py] = toTexProjector(lat, lon);
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    hasOutline = true;
  }
  if (hasOutline) {
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  // 6. Split divider + BASELINE | SHOCK labels on 3D surface
  if (viewMode === 'SPLIT') {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.72)';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(TEX_W / 2, 0);
    ctx.lineTo(TEX_W / 2, TEX_H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '600 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';
    const labelY = Math.round(TEX_H * 0.14);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
    ctx.fillRect(TEX_W / 2 - 86, labelY - 12, 76, 24);
    ctx.fillRect(TEX_W / 2 + 10, labelY - 12, 66, 24);
    ctx.fillStyle = 'rgba(212, 212, 216, 0.95)';
    ctx.fillText('BASELINE', TEX_W / 2 - 78, labelY);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText('SHOCK', TEX_W / 2 + 18, labelY);
    ctx.restore();
  }
}

export const Terrain3DView: React.FC<Terrain3DViewProps> = ({
  active,
  staticFields,
  reliefImage,
  showRelief,
  fieldImages,
  smooth,
  contourLayers,
  showGridLines,
  viewMode,
  selectedChannel,
  windMode,
  windFields,
  activePreset,
  customSpec,
  isBrushMode,
  onHoverLatLon,
  onBrushLatLon,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const staticDrapedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const windCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const textureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const textureRef = useRef<THREE.CanvasTexture | null>(null);
  const windFlowRef = useRef<WindFlow | null>(null);
  const terrainMeshRef = useRef<THREE.Mesh | null>(null);
  const startRafRef = useRef<(() => void) | null>(null);
  const enter3DRef = useRef<(() => void) | null>(null);
  const exit3DRef = useRef<(() => void) | null>(null);
  const activeRef = useRef(active);
  const [esriBasemap, setEsriBasemap] = useState<HTMLCanvasElement | null>(null);

  const hoverCbRef = useRef(onHoverLatLon);
  const brushCbRef = useRef(onBrushLatLon);
  const brushModeRef = useRef(isBrushMode);
  const windConfigRef = useRef({
    windMode,
    windFields,
    viewMode,
    selectedChannel,
  });

  useEffect(() => {
    activeRef.current = active;
    hoverCbRef.current = onHoverLatLon;
    brushCbRef.current = onBrushLatLon;
    brushModeRef.current = isBrushMode;
    windConfigRef.current = { windMode, windFields, viewMode, selectedChannel };
  }, [active, onHoverLatLon, onBrushLatLon, isBrushMode, windMode, windFields, viewMode, selectedChannel]);

  // Load Esri World Dark Gray tiles once (with CORS) for crisp borders, coastlines & labels
  useEffect(() => {
    let alive = true;
    loadEsriDomainBasemap().then((canvas) => {
      if (alive && canvas) setEsriBasemap(canvas);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Precompute terrain base canvas: Esri basemap + ocean/land contrast + crisp coastline contour
  const baseTerrainCanvas = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = TEX_W;
    canvas.height = TEX_H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

    if (esriBasemap) {
      ctx.drawImage(esriBasemap, 0, 0);
      // Subtle ocean-vs-land contrast overlay so sea level reads dark & crisp against landmass
      const shade = document.createElement('canvas');
      shade.width = NLON * 4;
      shade.height = NLAT * 4;
      const shCtx = shade.getContext('2d');
      if (shCtx) {
        const img = shCtx.createImageData(shade.width, shade.height);
        const d = img.data;
        for (let py = 0; py < shade.height; py++) {
          const fi = NLAT - (py + 0.5) / 4 - 0.5;
          for (let px = 0; px < shade.width; px++) {
            const fj = (px + 0.5) / 4 - 0.5;
            const lsm = sampleBilinear(staticFields.landSea, fi, fj);
            const k = (py * shade.width + px) * 4;
            if (lsm < 0.45) {
              d[k] = 4;
              d[k + 1] = 8;
              d[k + 2] = 16;
              d[k + 3] = Math.round((1 - lsm / 0.45) * 75);
            } else {
              d[k + 3] = 0;
            }
          }
        }
        shCtx.putImageData(img, 0, 0);
        ctx.drawImage(shade, 0, 0, TEX_W, TEX_H);
      }
    } else {
      // Offline fallback if Esri tiles unreachable: clean obsidian ocean + slate land relief
      const img = ctx.createImageData(TEX_W, TEX_H);
      const d = img.data;
      for (let py = 0; py < TEX_H; py++) {
        const fi = NLAT - (py + 0.5) / TEX_S - 0.5;
        for (let px = 0; px < TEX_W; px++) {
          const fj = (px + 0.5) / TEX_S - 0.5;
          const lsm = sampleBilinear(staticFields.landSea, fi, fj);
          const elev = Math.max(0, sampleBilinear(staticFields.orography, fi, fj));
          const isLand = lsm >= 0.4;
          const elevNorm = Math.min(1, elev / MAX_OROGRAPHY_M);
          const k = (py * TEX_W + px) * 4;
          if (!isLand) {
            d[k] = 7;
            d[k + 1] = 11;
            d[k + 2] = 18;
            d[k + 3] = 255;
          } else {
            const tone = 24 + Math.round(elevNorm * 42);
            d[k] = tone;
            d[k + 1] = tone + 5;
            d[k + 2] = tone + 12;
            d[k + 3] = 255;
          }
        }
      }
      ctx.putImageData(img, 0, 0);
    }

    // Crisp coastline contour (land_sea_mask = 0.5)
    const coastSegs = contourSegments(staticFields.landSea, 0.5, 4);
    if (coastSegs.length > 0) {
      ctx.beginPath();
      for (const [i1, j1, i2, j2] of coastSegs) {
        ctx.moveTo((j1 + 0.5) * TEX_S, (NLAT - 0.5 - i1) * TEX_S);
        ctx.lineTo((j2 + 0.5) * TEX_S, (NLAT - 0.5 - i2) * TEX_S);
      }
      ctx.strokeStyle = esriBasemap ? 'rgba(228, 228, 231, 0.35)' : 'rgba(228, 228, 231, 0.58)';
      ctx.lineWidth = 1.4;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    return canvas;
  }, [staticFields, esriBasemap]);

  // Build 3D Three.js scene, 96x108 heightfield, lighting, wind loop, and orbit controls
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.0085);

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.5, 220);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    // Ridge-enhancing directional lighting + clean ambient fill
    const ambient = new THREE.AmbientLight(0xffffff, 0.64);
    scene.add(ambient);

    const sunNW = new THREE.DirectionalLight(0xf8fafc, 1.75);
    sunNW.position.set(-22, 26, -16);
    scene.add(sunNW);

    const rimSE = new THREE.DirectionalLight(0x94a3b8, 0.48);
    rimSE.position.set(16, 9, 20);
    scene.add(rimSE);

    // Bicubic 160x180 sculptural heightfield mesh (flat ocean at z=0)
    const geometry = new THREE.PlaneGeometry(WORLD_W, WORLD_D, SEG_X, SEG_Z);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    for (let iz = 0; iz <= SEG_Z; iz++) {
      const fi = (1 - iz / SEG_Z) * NLAT - 0.5;
      for (let ix = 0; ix <= SEG_X; ix++) {
        const fj = (ix / SEG_X) * NLON - 0.5;
        const idx = iz * (SEG_X + 1) + ix;
        const elev = sampleBicubic(staticFields.orography, fi, fj);
        const lsm = sampleBilinear(staticFields.landSea, fi, fj);
        const y = lsm < 0.15 && elev < 20 ? 0 : (elev / MAX_OROGRAPHY_M) * MAX_HEIGHT;
        pos.setY(idx, y);
      }
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();

    if (!staticDrapedCanvasRef.current) {
      const sc = document.createElement('canvas');
      sc.width = TEX_W;
      sc.height = TEX_H;
      staticDrapedCanvasRef.current = sc;
    }
    if (!windCanvasRef.current) {
      const wc = document.createElement('canvas');
      wc.width = TEX_W;
      wc.height = TEX_H;
      windCanvasRef.current = wc;
    }
    if (!textureCanvasRef.current) {
      const tc = document.createElement('canvas');
      tc.width = TEX_W;
      tc.height = TEX_H;
      textureCanvasRef.current = tc;
    }

    const texture = new THREE.CanvasTexture(textureCanvasRef.current);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    textureRef.current = texture;

    const material = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.84,
      metalness: 0.05,
      side: THREE.DoubleSide,
    });

    const terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.scale.y = 0.001;
    terrainMeshRef.current = terrainMesh;
    scene.add(terrainMesh);

    const borderGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(WORLD_W, 0.08, WORLD_D));
    const borderMat = new THREE.LineBasicMaterial({ color: 0x3f3f46, transparent: true, opacity: 0.65 });
    const borderWire = new THREE.LineSegments(borderGeo, borderMat);
    borderWire.position.y = -0.04;
    scene.add(borderWire);

    if (!windFlowRef.current) windFlowRef.current = new WindFlow(1400);

    // Framed cleanly between the left and right glass side panels
    const target = new THREE.Vector3(0, 0.8, 0.4);
    let theta = 0.0;
    let phi = 0.08;
    let radius = 44;
    let targetTheta = 0.07;
    let targetPhi = 0.88;
    let targetRadius = 47;
    let enterStartMs = 0;
    let exitStartMs = 0;
    let exitFromTheta = 0.07;
    let exitFromPhi = 0.88;
    let exitFromRadius = 47;
    let exitFromScaleY = 1.0;
    const TRANSITION_DURATION_MS = 360;

    enter3DRef.current = () => {
      // Reset to near-top-down flat plane, then ease smoothly into 3D orbit + extruded relief
      exitStartMs = 0;
      theta = 0.0;
      phi = 0.08;
      radius = 44;
      targetTheta = 0.07;
      targetPhi = 0.88;
      targetRadius = 47;
      terrainMesh.scale.y = 0.001;
      enterStartMs = performance.now();
    };

    exit3DRef.current = () => {
      // Capture current 3D pose & ease smoothly back to flat top-down while crossfading to 2D
      enterStartMs = 0;
      exitFromTheta = theta;
      exitFromPhi = phi;
      exitFromRadius = radius;
      exitFromScaleY = terrainMesh.scale.y;
      exitStartMs = performance.now();
    };

    let isDragging = false;
    let pointerDownX = 0;
    let pointerDownY = 0;
    let lastX = 0;
    let lastY = 0;
    let totalDragPx = 0;

    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();

    const raycastTerrain = (clientX: number, clientY: number) => {
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObject(terrainMesh, false);
      if (hits.length === 0) return null;
      return worldToLatLon(hits[0].point.x, hits[0].point.z);
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      isDragging = true;
      pointerDownX = e.clientX;
      pointerDownY = e.clientY;
      lastX = e.clientX;
      lastY = e.clientY;
      totalDragPx = 0;
      renderer.domElement.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (isDragging) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        totalDragPx += Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY);
        targetTheta -= dx * 0.0065;
        targetPhi = Math.max(0.2, Math.min(1.42, targetPhi - dy * 0.0065));
        hoverCbRef.current(null, null);
        return;
      }
      const ll = raycastTerrain(e.clientX, e.clientY);
      if (ll) hoverCbRef.current(ll.lat, ll.lon);
      else hoverCbRef.current(null, null);
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging) return;
      isDragging = false;
      if (renderer.domElement.hasPointerCapture(e.pointerId)) {
        renderer.domElement.releasePointerCapture(e.pointerId);
      }
      if (totalDragPx < 6 && brushModeRef.current) {
        const ll = raycastTerrain(e.clientX, e.clientY);
        if (ll) brushCbRef.current(ll.lat, ll.lon);
      }
    };

    const onPointerLeave = () => {
      if (!isDragging) hoverCbRef.current(null, null);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      targetRadius = Math.max(16, Math.min(64, targetRadius + e.deltaY * 0.028));
    };

    const el = renderer.domElement;
    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', onPointerUp);
    el.addEventListener('pointerleave', onPointerLeave);
    el.addEventListener('wheel', onWheel, { passive: false });

    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    let raf = 0;
    let fadeOutUntilMs = 0;
    const animate = () => {
      if (!activeRef.current && performance.now() > fadeOutUntilMs) {
        raf = 0;
        return;
      }

      // Step live wind particles onto windCanvas when in animated FLOW mode
      const { windMode: wm, windFields: wf, viewMode: vm, selectedChannel: ch } = windConfigRef.current;
      const sc = staticDrapedCanvasRef.current;
      const wc = windCanvasRef.current;
      const tc = textureCanvasRef.current;
      const isAnom = vm === 'ANOMALY_DIFF';
      if (wm === 'FLOW' && !prefersReducedMotion() && !(isAnom && wf.deltaPeak <= 1e-3) && sc && wc && tc) {
        const wctx = wc.getContext('2d');
        const tctx = tc.getContext('2d');
        if (wctx && tctx && windFlowRef.current) {
          wctx.save();
          wctx.globalCompositeOperation = 'destination-in';
          wctx.fillStyle = 'rgba(0, 0, 0, 0.90)';
          wctx.fillRect(0, 0, TEX_W, TEX_H);
          wctx.restore();

          const colourBySpeed = isWindChannel(ch) && !isAnom;
          const fieldAt = (x: number): WindField => {
            if (isAnom) return wf.delta;
            if (vm === 'BASELINE') return wf.base;
            if (vm === 'SPLIT') return x < TEX_W / 2 ? wf.base : wf.shock;
            return wf.shock;
          };
          windFlowRef.current.step(wctx, toTexProjector, fieldAt, colourBySpeed);

          tctx.clearRect(0, 0, TEX_W, TEX_H);
          tctx.drawImage(sc, 0, 0);
          tctx.drawImage(wc, 0, 0);
          texture.needsUpdate = true;
        }
      }

      // Symmetric smooth pitch & elevation ease-in (2D->3D) and ease-out (3D->2D) (~360ms)
      if (enterStartMs > 0) {
        const t = Math.min(1, (performance.now() - enterStartMs) / TRANSITION_DURATION_MS);
        const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
        terrainMesh.scale.y = Math.max(0.001, ease);
        theta = 0.0 + (targetTheta - 0.0) * ease;
        phi = 0.08 + (targetPhi - 0.08) * ease;
        radius = 44 + (targetRadius - 44) * ease;
        if (t >= 1) enterStartMs = 0;
      } else if (exitStartMs > 0) {
        const t = Math.min(1, (performance.now() - exitStartMs) / TRANSITION_DURATION_MS);
        const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
        terrainMesh.scale.y = Math.max(0.001, exitFromScaleY + (0.001 - exitFromScaleY) * ease);
        theta = exitFromTheta + (0.0 - exitFromTheta) * ease;
        phi = exitFromPhi + (0.08 - exitFromPhi) * ease;
        radius = exitFromRadius + (44 - exitFromRadius) * ease;
        if (t >= 1) exitStartMs = 0;
      } else {
        theta += (targetTheta - theta) * 0.14;
        phi += (targetPhi - phi) * 0.14;
        radius += (targetRadius - radius) * 0.14;
      }

      camera.position.set(
        target.x + radius * Math.sin(phi) * Math.sin(theta),
        target.y + radius * Math.cos(phi),
        target.z + radius * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(target);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };

    startRafRef.current = () => {
      if (!activeRef.current) {
        fadeOutUntilMs = performance.now() + TRANSITION_DURATION_MS + 20;
      }
      if (!raf) {
        raf = requestAnimationFrame(animate);
      }
    };

    if (activeRef.current) {
      enter3DRef.current();
      raf = requestAnimationFrame(animate);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      startRafRef.current = null;
      enter3DRef.current = null;
      exit3DRef.current = null;
      terrainMeshRef.current = null;
      resizeObserver.disconnect();
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerUp);
      el.removeEventListener('pointerleave', onPointerLeave);
      el.removeEventListener('wheel', onWheel);
      if (el.parentNode === container) container.removeChild(el);
      geometry.dispose();
      material.dispose();
      texture.dispose();
      borderGeo.dispose();
      borderMat.dispose();
      renderer.dispose();
      textureRef.current = null;
    };
  }, [staticFields]);

  // Start/resume RAF and ease-in/ease-out pitch + elevation when active toggles
  const wasActiveRef = useRef(active);
  useEffect(() => {
    if (active) {
      enter3DRef.current?.();
      startRafRef.current?.();
    } else if (wasActiveRef.current) {
      exit3DRef.current?.();
      startRafRef.current?.(); // runs for 360ms exit ease-out + fade-out then pauses
    }
    wasActiveRef.current = active;
  }, [active]);

  // Clear stale wind trails immediately when switching viewMode (e.g. BASELINE <-> SHOCK <-> SPLIT)
  useEffect(() => {
    const wc = windCanvasRef.current;
    if (!wc) return;
    const wctx = wc.getContext('2d');
    wctx?.clearRect(0, 0, TEX_W, TEX_H);
  }, [viewMode]);

  // Redraw static draped layer (basemap + relief + field + contours + shock) and static wind arrows if active
  useEffect(() => {
    if (!active) return;
    const sc = staticDrapedCanvasRef.current;
    const wc = windCanvasRef.current;
    const tc = textureCanvasRef.current;
    const tex = textureRef.current;
    if (!sc || !wc || !tc || !tex) return;

    drawStaticDrapedCanvas(
      sc,
      baseTerrainCanvas,
      reliefImage,
      showRelief,
      fieldImages,
      smooth,
      contourLayers,
      showGridLines,
      viewMode,
      activePreset,
      customSpec,
      isBrushMode
    );

    const wctx = wc.getContext('2d');
    const isAnom = viewMode === 'ANOMALY_DIFF';
    if (wctx) {
      if (windMode === 'OFF' || (isAnom && windFields.deltaPeak <= 1e-3)) {
        wctx.clearRect(0, 0, TEX_W, TEX_H);
      } else if (windMode === 'ARROWS' || prefersReducedMotion()) {
        wctx.clearRect(0, 0, TEX_W, TEX_H);
        const colourBySpeed = isWindChannel(selectedChannel) && !isAnom;
        const fieldAt = (x: number): WindField => {
          if (isAnom) return windFields.delta;
          if (viewMode === 'BASELINE') return windFields.base;
          if (viewMode === 'SPLIT') return x < TEX_W / 2 ? windFields.base : windFields.shock;
          return windFields.shock;
        };
        drawWindArrows(wctx, toTexProjector, fieldAt, colourBySpeed, 2);
      }
    }

    const tctx = tc.getContext('2d');
    if (tctx) {
      tctx.clearRect(0, 0, TEX_W, TEX_H);
      tctx.drawImage(sc, 0, 0);
      tctx.drawImage(wc, 0, 0);
      tex.needsUpdate = true;
    }
  }, [
    active,
    baseTerrainCanvas,
    reliefImage,
    showRelief,
    fieldImages,
    smooth,
    contourLayers,
    showGridLines,
    viewMode,
    selectedChannel,
    windMode,
    windFields,
    activePreset,
    customSpec,
    isBrushMode,
  ]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 z-[14] bg-[#050505] transition-opacity duration-[360ms] ease-out ${
        active ? 'opacity-100' : 'opacity-0 pointer-events-none'
      } ${isBrushMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'}`}
    />
  );
};
