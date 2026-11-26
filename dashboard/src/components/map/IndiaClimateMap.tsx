/**
 * IndiaClimateMap.tsx - Leaflet 32x36 India Domain Climate Map Viewer
 * Bounds: 6.5°N - 37.5°N, 65.5°E - 100.5°E
 * Supports Baseline, Shock Multiverse, Dual Split-Screen, and Anomaly Diff views.
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import L from 'leaflet';
import { GRID_CONFIG, CHANNELS } from '../../types/climate';
import { useClimateData } from '../../context/ClimateDataContext';
import { useTimeline } from '../../context/TimelineContext';
import { useShockLab } from '../../context/ShockLabContext';
import { getColorForChannel, getAnomalyDivergentColor, rgbToCss } from '../../services/colormaps';
import { latLonToGrid, wireToDisplay, displayDeltaString } from '../../services/physicalUnits';
import { ColorbarLegend } from './ColorbarLegend';
import { ChannelSelector } from './ChannelSelector';
import { Crosshair, Layers } from 'lucide-react';

interface HoverInfo {
  lat: number;
  lon: number;
  gridI: number;
  gridJ: number;
  baseVal: number;
  shockVal: number;
  delta: number;
}

export const IndiaClimateMap: React.FC = () => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const shockCircleRef = useRef<L.Circle | null>(null);

  const {
    baselineRollout,
    shockRollout,
    initialCondition,
    selectedChannel,
    viewMode,
  } = useClimateData();

  const { currentDay } = useTimeline();
  const { customSpec, updateCustomSpec, isBrushMode } = useShockLab();

  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [showGridLines, setShowGridLines] = useState<boolean>(false);

  // Active channel index
  const chIdx = useMemo(() => CHANNELS[selectedChannel].index, [selectedChannel]);

  // Active grid data for current lead time
  const currentBaselineGrid = useMemo(() => {
    if (currentDay === 0) return initialCondition[chIdx];
    const frame = baselineRollout[currentDay - 1];
    return frame ? frame[chIdx] : initialCondition[chIdx];
  }, [currentDay, baselineRollout, initialCondition, chIdx]);

  const currentShockGrid = useMemo(() => {
    if (currentDay === 0) return initialCondition[chIdx];
    const frame = shockRollout[currentDay - 1];
    return frame ? frame[chIdx] : initialCondition[chIdx];
  }, [currentDay, shockRollout, initialCondition, chIdx]);

  // Max absolute anomaly for divergent scaling
  const maxAbsDelta = useMemo(() => {
    let maxD = 0.1;
    if (!currentBaselineGrid || !currentShockGrid) return maxD;
    for (let i = 0; i < GRID_CONFIG.nLat; i++) {
      for (let j = 0; j < GRID_CONFIG.nLon; j++) {
        const d = Math.abs(currentShockGrid[i][j] - currentBaselineGrid[i][j]);
        if (d > maxD) maxD = d;
      }
    }
    return Math.max(1.0, maxD);
  }, [currentBaselineGrid, currentShockGrid]);

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center over central India
    const map = L.map(mapContainerRef.current, {
      center: [21.5, 82.5],
      zoom: 5,
      minZoom: 4,
      maxZoom: 9,
      zoomControl: false,
      attributionControl: false,
    });

    // Dark high-contrast basemap (Esri World Dark Gray - Free public tier, NO API KEY required)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 16,
        attribution: 'Tiles &copy; Esri',
      }
    ).addTo(map);

    // Invalidate map size after layout stabilizes
    setTimeout(() => {
      map.invalidateSize();
    }, 150);

    // Bounding box boundary rectangle for India domain (6.5-37.5N, 65.5-100.5E)
    const bounds: L.LatLngBoundsExpression = [
      [GRID_CONFIG.latMin - 0.5, GRID_CONFIG.lonMin - 0.5],
      [GRID_CONFIG.latMax + 0.5, GRID_CONFIG.lonMax + 0.5],
    ];
    L.rectangle(bounds, {
      color: '#334155',
      weight: 1.5,
      fill: false,
      dashArray: '4, 4',
    }).addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Shock Brush visual preview circle on Leaflet
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (shockCircleRef.current) {
      shockCircleRef.current.remove();
      shockCircleRef.current = null;
    }

    if (isBrushMode || customSpec.magnitude !== 0) {
      const circle = L.circle([customSpec.centerLat, customSpec.centerLon], {
        radius: customSpec.radiusKm * 1000,
        color: '#0284C7',
        fillColor: '#0284C7',
        fillOpacity: 0.12,
        weight: 1.5,
        dashArray: '4, 4',
      }).addTo(map);

      shockCircleRef.current = circle;
    }
  }, [customSpec, isBrushMode]);

  // Map click handler for interactive shock painting
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const handleMapClick = (e: L.LeafletMouseEvent) => {
      if (!isBrushMode) return;
      const { lat, lng } = e.latlng;
      if (
        lat >= GRID_CONFIG.latMin &&
        lat <= GRID_CONFIG.latMax &&
        lng >= GRID_CONFIG.lonMin &&
        lng <= GRID_CONFIG.lonMax
      ) {
        updateCustomSpec({
          centerLat: Math.round(lat * 10) / 10,
          centerLon: Math.round(lng * 10) / 10,
        });
      }
    };

    map.on('click', handleMapClick);
    return () => {
      map.off('click', handleMapClick);
    };
  }, [isBrushMode, updateCustomSpec]);

  // Render 32x36 grid raster onto HTML5 canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const map = mapInstanceRef.current;
    if (!canvas || !map || !currentBaselineGrid || !currentShockGrid) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const splitX = width / 2;
      const cellHalfDeg = GRID_CONFIG.resolutionDeg / 2;

      for (let i = 0; i < GRID_CONFIG.nLat; i++) {
        const cellLat = GRID_CONFIG.latMin + i * GRID_CONFIG.resolutionDeg;

        for (let j = 0; j < GRID_CONFIG.nLon; j++) {
          const cellLon = GRID_CONFIG.lonMin + j * GRID_CONFIG.resolutionDeg;

          const nw = map.latLngToContainerPoint([cellLat + cellHalfDeg, cellLon - cellHalfDeg]);
          const se = map.latLngToContainerPoint([cellLat - cellHalfDeg, cellLon + cellHalfDeg]);

          const rectX = Math.round(nw.x);
          const rectY = Math.round(nw.y);
          const rectW = Math.round(se.x - nw.x);
          const rectH = Math.round(se.y - nw.y);

          const baseVal = currentBaselineGrid[i][j];
          const shockVal = currentShockGrid[i][j];
          const delta = shockVal - baseVal;

          let colorRgb;
          let alpha = 0.60;

          if (viewMode === 'ANOMALY_DIFF') {
            if (Math.abs(delta) < 0.05) {
              // Zero anomaly: keep completely transparent to expose underlying geography
              continue;
            }
            colorRgb = getAnomalyDivergentColor(delta, maxAbsDelta);
            alpha = Math.min(0.85, 0.35 + (Math.abs(delta) / maxAbsDelta) * 0.50);
          } else if (viewMode === 'BASELINE') {
            if (selectedChannel === 'tp' && baseVal < 0.5) {
              // Dry weather (0 mm rain): transparent so India terrain/coastline is fully visible
              continue;
            }
            colorRgb = getColorForChannel(baseVal, selectedChannel);
            alpha = selectedChannel === 'tp' ? Math.min(0.85, 0.40 + (baseVal / 40) * 0.45) : 0.55;
          } else if (viewMode === 'SPLIT') {
            const cellCenterX = rectX + rectW / 2;
            const isLeft = cellCenterX < splitX;
            const val = isLeft ? baseVal : shockVal;
            if (selectedChannel === 'tp' && val < 0.5) continue;
            colorRgb = getColorForChannel(val, selectedChannel);
            alpha = selectedChannel === 'tp' ? Math.min(0.85, 0.40 + (val / 40) * 0.45) : 0.55;
          } else {
            // SHOCK view
            if (selectedChannel === 'tp' && shockVal < 0.5) {
              // Dry weather (0 mm rain): transparent so India terrain/coastline is fully visible
              continue;
            }
            colorRgb = getColorForChannel(shockVal, selectedChannel);
            alpha = selectedChannel === 'tp' ? Math.min(0.85, 0.40 + (shockVal / 40) * 0.45) : 0.55;
          }

          ctx.fillStyle = rgbToCss(colorRgb, alpha);
          ctx.fillRect(rectX, rectY, rectW, rectH);

          if (showGridLines) {
            ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(rectX, rectY, rectW, rectH);
          }
        }
      }

      // Draw vertical divider if in SPLIT mode
      if (viewMode === 'SPLIT') {
        ctx.strokeStyle = '#0284C7';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(splitX, 0);
        ctx.lineTo(splitX, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };

    const handleResizeOrMove = () => {
      const size = map.getSize();
      if (size.x > 0 && size.y > 0) {
        canvas.width = size.x;
        canvas.height = size.y;
      }
      render();
    };

    handleResizeOrMove();
    map.on('move', render);
    map.on('zoom', handleResizeOrMove);
    map.on('resize', handleResizeOrMove);

    return () => {
      map.off('move', render);
      map.off('zoom', handleResizeOrMove);
      map.off('resize', handleResizeOrMove);
    };
  }, [
    currentBaselineGrid,
    currentShockGrid,
    viewMode,
    selectedChannel,
    maxAbsDelta,
    showGridLines,
  ]);

  // Mouse move handler for inspector readout
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const map = mapInstanceRef.current;
    if (!map || !currentBaselineGrid || !currentShockGrid) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const latLng = map.containerPointToLatLng([x, y]);
    const grid = latLonToGrid(latLng.lat, latLng.lng);

    if (grid) {
      const baseVal = currentBaselineGrid[grid.i][grid.j];
      const shockVal = currentShockGrid[grid.i][grid.j];
      setHoverInfo({
        lat: Math.round(latLng.lat * 10) / 10,
        lon: Math.round(latLng.lng * 10) / 10,
        gridI: grid.i,
        gridJ: grid.j,
        baseVal,
        shockVal,
        delta: shockVal - baseVal,
      });
    } else {
      setHoverInfo(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  return (
    <div
      className="relative w-full h-full bg-[#070B11] overflow-hidden select-none"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Leaflet DOM Node */}
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Meteorological Canvas Overlay */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none z-10"
      />

      {/* Split-Screen Header Banners if in SPLIT mode */}
      {viewMode === 'SPLIT' && (
        <div className="absolute top-14 inset-x-0 z-20 flex justify-between px-12 pointer-events-none text-xs font-mono">
          <div className="bg-[#111827]/90 border border-slate-700 px-3 py-1 rounded text-slate-300 shadow">
            ← BASELINE (P = 0)
          </div>
          <div className="bg-[#111827]/90 border border-sky-600 px-3 py-1 rounded text-sky-400 shadow">
            SHOCK MULTIVERSE →
          </div>
        </div>
      )}

      {/* Top Map Controls: Channel Selector & Layer Toggles */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
        <ChannelSelector />

        <div className="flex items-center bg-[#111827] border border-[#1E293B] rounded p-1 text-xs font-mono text-slate-400">
          <button
            onClick={() => setShowGridLines((prev) => !prev)}
            className={`px-2 py-1 rounded transition-colors flex items-center gap-1 ${
              showGridLines ? 'bg-[#1E293B] text-sky-400 font-semibold' : 'hover:text-slate-200'
            }`}
            title="Toggle 1.0° Grid Lines"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>GRID</span>
          </button>
        </div>
      </div>

      {/* Top Right: View Mode Badge */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
        <div className="px-2.5 py-1 rounded bg-[#111827]/90 border border-[#1E293B] text-xs font-mono text-slate-300">
          DOMAIN: <span className="text-sky-400">32x36 (1.0° res)</span>
        </div>
      </div>

      {/* Bottom Left: Quantitative Colorbar */}
      <div className="absolute bottom-4 left-4 z-20">
        <ColorbarLegend
          isAnomaly={viewMode === 'ANOMALY_DIFF'}
          maxAbsDelta={maxAbsDelta}
        />
      </div>

      {/* Bottom Right: Lat/Lon Grid Inspector Pill */}
      {hoverInfo && (
        <div className="absolute bottom-4 right-4 z-20 bg-[#111827]/95 border border-[#1E293B] rounded p-3 text-xs font-mono shadow-2xl tabular-nums">
          <div className="flex items-center gap-2 mb-1.5 text-slate-400 border-b border-slate-800 pb-1">
            <Crosshair className="w-3.5 h-3.5 text-sky-400" />
            <span>
              {hoverInfo.lat.toFixed(1)}°N, {hoverInfo.lon.toFixed(1)}°E
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-500">
              [{hoverInfo.gridI}, {hoverInfo.gridJ}]
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <span className="text-slate-500 block text-[10px]">BASELINE</span>
              <span className="text-slate-200 font-medium">
                {wireToDisplay(hoverInfo.baseVal, selectedChannel).str} {wireToDisplay(hoverInfo.baseVal, selectedChannel).unit}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">SHOCK</span>
              <span className="text-sky-400 font-medium">
                {wireToDisplay(hoverInfo.shockVal, selectedChannel).str} {wireToDisplay(hoverInfo.shockVal, selectedChannel).unit}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">ANOMALY Δ</span>
              <span
                className={`font-semibold ${
                  hoverInfo.delta > 0
                    ? 'text-red-400'
                    : hoverInfo.delta < 0
                    ? 'text-sky-400'
                    : 'text-slate-400'
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
