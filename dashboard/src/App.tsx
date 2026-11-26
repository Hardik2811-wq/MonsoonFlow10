/**
 * App.tsx - MonsoonFlow-10 Climate Digital Twin Dashboard
 * High-craft meteorological instrument console adhering strictly to unslop design standards.
 */

import { ShockLabProvider } from './context/ShockLabContext';
import { ClimateDataProvider } from './context/ClimateDataContext';
import { TimelineProvider } from './context/TimelineContext';
import { Header } from './components/layout/Header';
import { ShockLabPanel } from './components/multiverse/ShockLabPanel';
import { IndiaClimateMap } from './components/map/IndiaClimateMap';
import { DownstreamImpactPanel } from './components/downstream/DownstreamImpactPanel';
import { TimelineScrubber } from './components/timeline/TimelineScrubber';

export function App() {
  return (
    <ShockLabProvider>
      <ClimateDataProvider>
        <TimelineProvider>
          <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#0B0F17] text-slate-100 select-none">
            {/* Top Meteorological Console Header */}
            <Header />

            {/* Main Instrument Workspace: Shock Lab | India Climate Map | Downstream Impacts */}
            {/* The map runs full-width underneath; the side panels float over it as glass */}
            <div className="relative flex-1 overflow-hidden">
              <div className="absolute inset-0">
                <IndiaClimateMap />
              </div>

              {/* Left: Multiverse Counterfactual Shock Lab */}
              <div className="absolute left-3 top-3 bottom-3 z-30 flex">
                <ShockLabPanel />
              </div>

              {/* Right: Humanitarian & Agricultural Downstream Impact Module */}
              <div className="absolute right-3 top-3 bottom-3 z-30 flex">
                <DownstreamImpactPanel />
              </div>
            </div>

            {/* Bottom Deck: Forecast Timeline Scrubber (Day 0 - 10) */}
            <TimelineScrubber />
          </div>
        </TimelineProvider>
      </ClimateDataProvider>
    </ShockLabProvider>
  );
}

export default App;
