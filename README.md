# MonsoonFlow-10 (MF10) & Monsoon Multiverse

> **Event:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Repository:** [Hardik2811-wq/MonsoonFlow10](https://github.com/Hardik2811-wq/MonsoonFlow10)  
> **Architecture:** Compact Physics-Aware Spatiotemporal Climate Digital Twin & Counterfactual Shock Engine

---

## 1. Project Overview

**MonsoonFlow-10 (MF10)** is a lightweight, physics-constrained spatiotemporal forecasting engine and interactive digital twin designed for the Indian monsoon. It produces stable 10-day autoregressive rollouts across 6 dynamic atmospheric variables on a $32 \times 36$ regional grid while enabling interactive, localized counterfactual shock simulations.

### Key Capabilities
- **10-Day Spatiotemporal Rollout:** 10 forward simulation steps from 3 historical daily states ($t_{-2}, t_{-1}, t_0$) without runaway autoregressive drift.
- **Multiverse Shock Laboratory:** Interactive initial-condition perturbations ($X'_0 = X_0 + P$) propagating causally across lead days (including `warm_ocean` in Bay of Bengal and `monsoon_stall` in Arabian Sea).
- **Zero-Shock Consistency:** Exact baseline parity ($P = 0 \implies \text{atol} < 10^{-6}$) under deterministic CPU execution.
- **Offline Serving:** Runs completely offline (`--network none`) on 4 CPU cores, $\le 6$ GB RAM, with latency $\le 0.5$ s (well within the $5.0$ s competition ceiling).
- **Physical Validity:** Strictly non-negative precipitation ($tp \ge 0$), bounded temperature ($t2m \in [233.15, 333.15]$ K), and raw physical units on the wire.

---

## 2. Atmospheric & Spatial Domain

- **Grid Resolution:** $32 \times 36$ regular grid ($1^\circ \times 1^\circ$ resolution).
- **Spatial Coverage:** Latitude $6.5^\circ\text{N} \dots 37.5^\circ\text{N}$, Longitude $65.5^\circ\text{E} \dots 100.5^\circ\text{E}$.
- **6 Dynamic Channels (Physical Units):**
  1. `t2m`: 2-metre temperature ($K$)
  2. `tp`: Total precipitation ($24\text{ h}$ sum, $\text{mm/day} \ge 0$)
  3. `u10`: 10-metre zonal wind ($\text{m/s}$)
  4. `v10`: 10-metre meridional wind ($\text{m/s}$)
  5. `msl`: Mean sea-level pressure ($\text{Pa}$)
  6. `tcwv`: Total column water vapour ($\text{kg/m}^2 \ge 0$)
- **2 Static Terrain Channels:** `land_sea_mask` (binary) and `orography_m` (elevation in meters).

---

## 3. Quickstart & Serving

### 3.1 Offline CPU Inference Server (Root Entrypoint)
The submission executes directly from the repository root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start offline inference server on port 8000
python serve.py --port 8000
```

### 3.2 Automated Validation Gates
Run the official competition verification commands:

```bash
# Black-box API schema and latency validation
python -m monsoon_r1.schema_validator http://localhost:8000

# Static anti-cheat & code health audit
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"

# Evaluation scoring on Pool A (2019–2020)
python evaluation/eval_scoring.py
```

### 3.3 Interactive Digital Twin Dashboard
The frontend is decoupled in `dashboard/` (React 19 + TypeScript + Vite + Tailwind CSS + Leaflet):

```bash
cd dashboard
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to interact with the map, timeline scrubber, and shock lab.

---

## 4. Repository Structure

```text
MonsoonFlow10/
├── serve.py                     # ROOT ENTRYPOINT: Offline FastAPI inference server
├── requirements.txt             # Lean offline serving dependencies
├── .gitignore                   # Excludes raw .nc data, node_modules, internal logs
│
├── checkpoints/                 # Committed model weights & precomputed arrays
│   ├── stats.json               # Channel mean and std (2005–2018 training split)
│   ├── climatology.npy          # 366-day circularly smoothed seasonal baseline
│   └── static.npy               # 2x32x36 land_sea_mask and orography
│
├── guidelines/                  # Official DevForge Track 2 rules & guidelines
│   └── SUBMISSION_AND_REPOSITORY_GUIDELINES.md
│
├── plan/                        # Architecture & orchestration documentation
│   ├── 01_PROJECT_EXECUTION_PLAN.md
│   ├── 02_MULTI_AI_AGENTS_ORCHESTRATION_PLAN.md
│   └── 03_DEPENDENCIES_AND_SETUP_PLAN.md
│
├── data/                        # Climate data preprocessing & sliding-window dataset
├── inference/                   # CPU inference loader, predictor, and schemas
├── evaluation/                  # Baseline comparators & scoring wrappers
├── tests/                       # Pytest integrity & shock causality tests
└── dashboard/                   # Interactive digital twin web application
```

---

## 5. Submission Compliance & Guidelines
See [guidelines/SUBMISSION_AND_REPOSITORY_GUIDELINES.md](guidelines/SUBMISSION_AND_REPOSITORY_GUIDELINES.md) for full compliance details against pages 1, 8–10 of the DevForge brief.
