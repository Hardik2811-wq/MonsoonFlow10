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
          {/* The live map fills the whole screen; every piece of chrome floats over it as glass.
              Header = h-14 (56 px), timeline = h-[72px]; IndiaClimateMap keeps its overlays and
              domain fit clear of both (TOP_BAR_H / BOTTOM_BAR_H). */}
          <div className="relative h-screen w-screen overflow-hidden bg-[#050505] text-zinc-100 select-none">
            <div className="absolute inset-0">
              <IndiaClimateMap />
            </div>

            {/* Top Meteorological Console Header */}
            <div className="absolute top-0 inset-x-0 z-40">
              <Header />
            </div>

            {/* Left: Multiverse Counterfactual Shock Lab */}
            <div className="absolute left-3 top-[68px] bottom-[84px] z-30 flex">
              <ShockLabPanel />
            </div>

            {/* Right: Humanitarian & Agricultural Downstream Impact Module */}
            <div className="absolute right-3 top-[68px] bottom-[84px] z-30 flex">
              <DownstreamImpactPanel />
            </div>

            {/* Bottom Deck: Forecast Timeline Scrubber (Day 0 - 10) */}
            <div className="absolute bottom-0 inset-x-0 z-40">
              <TimelineScrubber />
            </div>
          </div>
        </TimelineProvider>
      </ClimateDataProvider>
    </ShockLabProvider>
  );
}

export default App;
