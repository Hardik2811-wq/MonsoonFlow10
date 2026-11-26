/**
 * Terrain3DView.tsx - True 3D terrain heightfield using real orography_m & land_sea_mask
 *
 * Upsamples the 32x36 orography_m grid to 96x108 with bilinear interpolation, extrudes real Z
 * elevation (Himalayas ~5,284 m, Tibetan Plateau, Western Ghats, Deccan Plateau above z=0 sea),
 * and drapes the live weather field, contours, coastlines, and shock outlines onto the surface.
 */

import React, { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { GRID_CONFIG } from '../../types/climate';
import type { ViewMode } from '../../context/ClimateDataContext';
import type { StaticFields } from '../../services/staticFields';
import type { GaussianShockSpec, ShockPreset } from '../../types/shocks';
import {
  contourSegments,
  sampleBilinear,
  type ContourStyle,
  type Segment,
} from '../../services/fieldRaster';

export interface TerrainContourLayer {
  segs: Segment[];
  style: ContourStyle;
  side: 'all' | 'left' | 'right';
}

interface Terrain3DViewProps {
  staticFields: StaticFields;
  fieldImages: {
    left: HTMLCanvasElement;
    right: HTMLCanvasElement | null;
    margin: number;
  };
  smooth: boolean;
  contourLayers: TerrainContourLayer[];
  showGridLines: boolean;
  viewMode: ViewMode;
  activePreset: ShockPreset;
  customSpec: GaussianShockSpec;
  isBrushMode: boolean;
  onHoverLatLon: (lat: number | null, lon: number | null) => void;
  onBrushLatLon: (lat: number, lon: number) => void;
}

const NLAT = GRID_CONFIG.nLat; // 32
const NLON = GRID_CONFIG.nLon; // 36
const HALF = GRID_CONFIG.resolutionDeg / 2; // 0.5

/** 3x bilinear upsampling -> 96x108 terrain mesh */
const SEG_X = 108;
const SEG_Z = 96;

/** World dimensions preserving cos(22°) aspect ratio of the 32°x36° India domain */
const WORLD_W = NLON * Math.cos((22 * Math.PI) / 180); // ~33.38
const WORLD_D = NLAT; // 32
const MAX_OROGRAPHY_M = 5284;
const MAX_HEIGHT = 5.4;

/** Draped texture resolution (16 px per 1° cell = 576x512) */
const TEX_S = 16;
const TEX_W = NLON * TEX_S;
const TEX_H = NLAT * TEX_S;

function worldToLatLon(x: number, z: number): { lat: number; lon: number } {
  const u = x / WORLD_W + 0.5; // 0..1 west->east
  const v = z / WORLD_D + 0.5; // 0..1 north->south
  const lon = GRID_CONFIG.lonMin - HALF + u * NLON;
  const lat = GRID_CONFIG.latMax + HALF - v * NLAT;
  return { lat, lon };
}

function drawDrapedTexture(
  canvas: HTMLCanvasElement,
  baseCanvas: HTMLCanvasElement,
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
  // 1. Pre-rendered land/ocean + coastline base
  ctx.drawImage(baseCanvas, 0, 0);

  // 2. Draped weather field (crop out any outer fade margin so it aligns 1:1 with the 32x36 mesh)
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

  // Linear projection from fractional grid coords (fi, fj) -> texture px
  const toTex = (fi: number, fj: number): [number, number] => [
    (fj + 0.5) * TEX_S,
    (NLAT - 0.5 - fi) * TEX_S,
  ];
  const latLonToTex = (lat: number, lon: number): [number, number] =>
    toTex(lat - GRID_CONFIG.latMin, lon - GRID_CONFIG.lonMin);

  // 3. Contours
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
    ctx.lineWidth = layer.style.width * 1.1;
    if (layer.style.glow > 0) {
      ctx.shadowColor = layer.style.color;
      ctx.shadowBlur = layer.style.glow;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = Math.max(0.7, layer.style.width * 0.5);
    }
    ctx.stroke();
    ctx.restore();
  }

  // 4. Grid lines
  if (showGridLines) {
    ctx.save();
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.lineWidth = 0.75;
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

  // 5. Active shock footprint draped onto terrain
  ctx.save();
  ctx.strokeStyle = '#38BDF8';
  ctx.fillStyle = 'rgba(2, 132, 199, 0.14)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  let hasOutline = false;
  if (activePreset.box) {
    const [la0, la1, lo0, lo1] = activePreset.box.box;
    const latLo = Math.ceil(la0 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin - HALF;
    const latHi = Math.floor(la1 - GRID_CONFIG.latMin) + GRID_CONFIG.latMin + HALF;
    const lonLo = Math.ceil(lo0 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin - HALF;
    const lonHi = Math.floor(lo1 - GRID_CONFIG.lonMin) + GRID_CONFIG.lonMin + HALF;
    const [bx0, by0] = latLonToTex(latHi, lonLo);
    const [bx1, by1] = latLonToTex(latLo, lonHi);
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
      const [px, py] = latLonToTex(lat, lon);
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

  // 6. Split divider
  if (viewMode === 'SPLIT') {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(TEX_W / 2, 0);
    ctx.lineTo(TEX_W / 2, TEX_H);
    ctx.stroke();
    ctx.restore();
  }
}

export const Terrain3DView: React.FC<Terrain3DViewProps> = ({
  staticFields,
  fieldImages,
  smooth,
  contourLayers,
  showGridLines,
  viewMode,
  activePreset,
  customSpec,
  isBrushMode,
  onHoverLatLon,
  onBrushLatLon,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const textureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const textureRef = useRef<THREE.CanvasTexture | null>(null);

  const hoverCbRef = useRef(onHoverLatLon);
  const brushCbRef = useRef(onBrushLatLon);
  const brushModeRef = useRef(isBrushMode);
  useEffect(() => {
    hoverCbRef.current = onHoverLatLon;
    brushCbRef.current = onBrushLatLon;
    brushModeRef.current = isBrushMode;
  }, [onHoverLatLon, onBrushLatLon, isBrushMode]);

  // Precompute terrain base canvas (land/ocean shading + crisp 0.5 coastline contour)
  const baseTerrainCanvas = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = TEX_W;
    canvas.height = TEX_H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

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
          d[k + 1] = 10;
          d[k + 2] = 15;
          d[k + 3] = 255;
        } else {
          // Dark slate landmass rising subtly in brightness toward the Tibetan Plateau
          const tone = 22 + Math.round(elevNorm * 38);
          d[k] = tone;
          d[k + 1] = tone + 5;
          d[k + 2] = tone + 12;
          d[k + 3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    // Crisp coastline contour (land_sea_mask = 0.5)
    const coastSegs = contourSegments(staticFields.landSea, 0.5, 4);
    if (coastSegs.length > 0) {
      ctx.beginPath();
      for (const [i1, j1, i2, j2] of coastSegs) {
        ctx.moveTo((j1 + 0.5) * TEX_S, (NLAT - 0.5 - i1) * TEX_S);
        ctx.lineTo((j2 + 0.5) * TEX_S, (NLAT - 0.5 - i2) * TEX_S);
      }
      ctx.strokeStyle = 'rgba(228, 228, 231, 0.5)';
      ctx.lineWidth = 1.3;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    return canvas;
  }, [staticFields]);

  // Build 3D Three.js scene, heightfield geometry, lighting, and orbit controls
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.011);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.5, 200);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    // Lighting for realistic 3D mountain relief
    const ambient = new THREE.AmbientLight(0xffffff, 0.62);
    scene.add(ambient);

    const sunNW = new THREE.DirectionalLight(0xf8fafc, 1.35);
    sunNW.position.set(-18, 24, -14);
    scene.add(sunNW);

    const rimSE = new THREE.DirectionalLight(0x94a3b8, 0.35);
    rimSE.position.set(14, 10, 18);
    scene.add(rimSE);

    // Bilinearly upsampled 96x108 heightfield mesh
    const geometry = new THREE.PlaneGeometry(WORLD_W, WORLD_D, SEG_X, SEG_Z);
    geometry.rotateX(-Math.PI / 2);

    const pos = geometry.attributes.position;
    for (let iz = 0; iz <= SEG_Z; iz++) {
      const fi = (1 - iz / SEG_Z) * NLAT - 0.5;
      for (let ix = 0; ix <= SEG_X; ix++) {
        const fj = (ix / SEG_X) * NLON - 0.5;
        const idx = iz * (SEG_X + 1) + ix;
        const elev = Math.max(0, sampleBilinear(staticFields.orography, fi, fj));
        const lsm = sampleBilinear(staticFields.landSea, fi, fj);
        const y = lsm < 0.15 && elev < 20 ? 0 : (elev / MAX_OROGRAPHY_M) * MAX_HEIGHT;
        pos.setY(idx, y);
      }
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();

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
      roughness: 0.88,
      metalness: 0.06,
      side: THREE.DoubleSide,
    });

    const terrainMesh = new THREE.Mesh(geometry, material);
    scene.add(terrainMesh);

    // Subtle domain plinth wireframe border at z = 0
    const borderGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(WORLD_W, 0.08, WORLD_D));
    const borderMat = new THREE.LineBasicMaterial({ color: 0x3f3f46, transparent: true, opacity: 0.6 });
    const borderWire = new THREE.LineSegments(borderGeo, borderMat);
    borderWire.position.y = -0.04;
    scene.add(borderWire);

    // Spherical orbit state (smoothly damped)
    const target = new THREE.Vector3(0, 0.9, 0);
    let theta = 0.06; // azimuth (rad), 0 = looking north from south
    let phi = 0.86; // polar angle from +Y (rad)
    let radius = 37;
    let targetTheta = theta;
    let targetPhi = phi;
    let targetRadius = radius;

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
      targetRadius = Math.max(15, Math.min(58, targetRadius + e.deltaY * 0.025));
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
    const animate = () => {
      theta += (targetTheta - theta) * 0.14;
      phi += (targetPhi - phi) * 0.14;
      radius += (targetRadius - radius) * 0.14;

      camera.position.set(
        target.x + radius * Math.sin(phi) * Math.sin(theta),
        target.y + radius * Math.cos(phi),
        target.z + radius * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(target);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
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

  // Update draped texture whenever field, contours, grid, view mode, or shock changes
  useEffect(() => {
    const tc = textureCanvasRef.current;
    const tex = textureRef.current;
    if (!tc || !tex) return;
    drawDrapedTexture(
      tc,
      baseTerrainCanvas,
      fieldImages,
      smooth,
      contourLayers,
      showGridLines,
      viewMode,
      activePreset,
      customSpec,
      isBrushMode
    );
    tex.needsUpdate = true;
  }, [
    baseTerrainCanvas,
    fieldImages,
    smooth,
    contourLayers,
    showGridLines,
    viewMode,
    activePreset,
    customSpec,
    isBrushMode,
  ]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 z-[14] bg-[#050505] ${isBrushMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'}`}
    />
  );
};
