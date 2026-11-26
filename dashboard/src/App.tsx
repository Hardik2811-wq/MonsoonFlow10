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
            <div className="flex flex-1 overflow-hidden">
              {/* Left Column: Multiverse Counterfactual Shock Lab */}
              <ShockLabPanel />

              {/* Center Column: Interactive 32x36 India Domain Climate Map */}
              <div className="flex-1 relative h-full">
                <IndiaClimateMap />
              </div>

              {/* Right Column: Humanitarian & Agricultural Downstream Impact Module */}
              <DownstreamImpactPanel />
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
