# MonsoonFlow-10 (MF10) & Monsoon Multiverse

> **Event:** Yugaantar DevForge 2026 — Track 2: Multiverse of Monsoon  
> **Repository:** [Hardik2811-wq/MonsoonFlow10](https://github.com/Hardik2811-wq/MonsoonFlow10)  
> **Core System:** Compact Physics-Aware Spatiotemporal Climate Digital Twin (`331,786` params, `1.30 MB`, `~28 ms` CPU 10-day rollout)

---

## 1. Executive Summary (For Invigilators)

**MonsoonFlow-10 (MF10)** is an offline, CPU-efficient spatiotemporal climate digital twin for the Indian monsoon. Given 3 historical daily states ($t_{-2}, t_{-1}, t_0$) across 6 atmospheric variables on a $32 \times 36$ regional grid, it produces a stable **10-day autoregressive rollout** `(10, 6, 32, 36)` and supports localized **counterfactual shock experiments** ($X'_0 = X_0 + P$) without runaway drift, teleportation, or external weather APIs.

### Verified Competition Metrics
| Evaluation Gate | Pool A (`2019–2020`, 50 windows) | Sealed Holdout (`2021`, 50 windows) | Target / Ceiling |
| :--- | :---: | :---: | :---: |
| **Total Predicted Score** | **`90.85 / 100.0`** | **`96.51 / 100.0`** | Higher is better |
| **Admission Gate (Beat Day-4 Persistence)** | **PASSED** (`42.844 < 61.177`) | **PASSED** (`46.706 < 65.125`) | Must beat persistence |
| **Forecast Skill (`/45`)** | `35.85 / 45.0` | `41.51 / 45.0` (CSI `15.0/15.0`) | `/45.0` |
| **Anti-Blur Variance Ratio (`/10`)** | **`10.0 / 10.0`** | **`10.0 / 10.0`** | `/10.0` |
| **Shock Causality (`/15`)** | **`15.0 / 15.0`** (`warm_ocean` 4/4, `monsoon_stall` 4/4) | **`15.0 / 15.0`** (`warm_ocean` 4/4, `monsoon_stall` 4/4) | `/15.0` |
| **Zero-Shock Parity ($P = 0$)** | `0.00e+00` max diff | `0.00e+00` max diff | `< 1e-6` |
| **Negative `tp` / `tcwv` Violations** | `0.00%` / `0.00%` | `0.00%` / `0.00%` | `0%` |
| **4-Core CPU 10-Day Latency** | **`25.3–28.2 ms`** mean (`32.6 ms` p95) | **`25.3–28.2 ms`** mean (`32.6 ms` p95) | `<= 5,000 ms` |
| **Checkpoint & Repo Footprint** | `1.30 MB` (`331,786` params) | `0.406 GB` total repo | `< 2.0 GB` |

---

## 2. Invigilator Quick-Verify (Offline CPU)

All weights (`checkpoints/model.pt`, `1.30 MB`), training statistics (`stats.json`), climatology (`climatology.npy`), and static terrain (`static.npy`) are committed in `checkpoints/`, so inference and API validation run immediately offline.

### Step 1 — Environment Setup
```bash
git clone https://github.com/Hardik2811-wq/MonsoonFlow10.git
cd MonsoonFlow10

python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate

pip install -r requirements.txt
# Optional official validator package:
pip install git+https://github.com/OfficialAbhinavSingh/yug-monsoon-kit.git
```

### Step 2 — Start the Root Inference Server (`--network none` ready)
```bash
python serve.py --port 8000
```
The server runs on 4 CPU threads (`torch.set_num_threads(4)`), loads `checkpoints/model.pt` with `weights_only=True`, and exposes `http://127.0.0.1:8000`.

### Step 3 — Run All Verification Gates (Second Terminal)
```bash
# 1. Unit & live-server shock causality tests (11/11 passed)
python -m pytest tests/ -v

# 2. Official black-box API schema, shock & 5.0s latency validator (6/6 passed)
python -m monsoon_r1.schema_validator http://127.0.0.1:8000

# 3. 4-core CPU latency & zero-shock parity benchmark
python inference/benchmark.py

# 4. Static anti-cheat, external-API & repository size audit
python evaluation/run_repo_scan.py
```

*(Note: Full-dataset scoring via `evaluation/eval_scoring.py` or `evaluation/eval_physics_audit.py` requires `india_climate_train.nc` in `data/` or `data/raw/`, excluded from git to keep the repository under 2 GB.)*

---

## 3. Model Architecture & Physics Design

MF10 uses **Model A (`exp4_msresunet_anchor`)** implemented in `models/backbones.py`:
- **Multi-Scale Residual U-Net (`MultiScaleResUNet`) + Seasonal Anchor (`SeasonalAnchorNet`):** Combines fine-scale spatial convolutions with a coarse synoptic branch (`base_channels=48`, `coarse_channels=72`) and day-of-year harmonic anchoring (`sin(2π·doy/366)`, `cos(2π·doy/366)`).
- **Static Terrain Conditioning:** Incorporates normalized `orography_m` (`orography_scale_m=8000.0`) and `land_sea_mask` so shocks and moisture transport respect Himalayan/Western Ghats topography and land–sea boundaries.
- **Physical Constraints & Tendency Rollout:** Predicts bounded daily tendencies in physical units with hard non-negativity on precipitation (`tp >= 0`) and column water vapour (`tcwv >= 0`), and physical bounds on 2 m temperature (`t2m in [233.15, 333.15]` K).
- **Causal Multiverse Perturbation (`X'_0 = X_0 + P`):** Localized Day-1 shocks (e.g., Bay of Bengal `warm_ocean`, Arabian Sea `monsoon_stall`, or custom Gaussian brush) stay confined near the source on Day 1 and advect/diffuse physically across Days 4–8 ($\le 8.0$ grid cells/day, zero teleportation).

---

## 4. Atmospheric Domain & HTTP API Contract

- **Grid:** $32 \times 36$ regular grid ($1^\circ \times 1^\circ$), Lat $6.5^\circ\text{N} \dots 37.5^\circ\text{N}$, Lon $65.5^\circ\text{E} \dots 100.5^\circ\text{E}$.
- **6 Dynamic Channels (Raw Physical Units):**
  1. `t2m`: 2-metre temperature ($\text{K}$)
  2. `tp`: Total precipitation ($24\text{ h}$ sum, $\text{mm/day} \ge 0$)
  3. `u10`: 10-metre zonal wind ($\text{m/s}$)
  4. `v10`: 10-metre meridional wind ($\text{m/s}$)
  5. `msl`: Mean sea-level pressure ($\text{Pa}$)
  6. `tcwv`: Total column water vapour ($\text{kg/m}^2 \ge 0$)
- **2 Static Channels:** `land_sea_mask` (binary), `orography_m` (meters).

### Endpoints (`serve.py`)
- `GET /health` — Server readiness, model name (`MonsoonFlow-10`), and device/thread info.
- `GET /climate/defaults` — Default 3-day initial state `(3, 6, 32, 36)`, `doy`, and static terrain arrays (`land_sea_mask`, `orography_m`) for the UI.
- `POST /simulate` — Accepts `initial_state` `(3, 6, 32, 36)`, optional `shock` `(6, 32, 36)`, `doy`, and `steps=10`; returns `baseline_forecast` and `perturbed_forecast` in `(10, 6, 32, 36)` raw physical units.

---

## 5. Anti-Leakage & Compliance Guarantees

1. **Strict Chronological Split (`data/preprocessing.py`):**
   - **Train:** `2005–2018` (normalization `stats.json` and 366-day `climatology.npy` computed strictly on `2005–2018`).
   - **Validation (Pool A):** `2019–2020`.
   - **Sealed Holdout:** `2021` (never used for training, normalization, or checkpoint selection).
2. **Offline & Clean:** Zero external weather packages (`open-meteo`, `cdsapi`, `weatherbench2`, etc.) and zero tracked NetCDF/binary datasets (`0.406 GB < 2.0 GB`).
3. **2-Page Brief:** See [`project_brief.pdf`](project_brief.pdf) for the concise technical summary.

---

## 6. Interactive Digital Twin Dashboard (`dashboard/`)

A decoupled React 19 + TypeScript + Vite + Three.js + Leaflet web application (`http://localhost:5173`) communicating live with `serve.py`:
```bash
cd dashboard
npm install
npm run dev
```
- **Symmetric 2D / 3D India Climate Map:** Seamless animated transition between 2D Leaflet geospatial heatmaps/wind vectors and a high-detail `256×288` Three.js 3D terrain relief mesh with identical overlays and interactive brush parity.
- **Monsoon Multiverse Shock Lab:** Apply preset shocks (`Warm Bay of Bengal`, `Monsoon Stall`) or paint localized Gaussian perturbations directly onto the map to compare Baseline vs. Perturbed vs. Anomaly trajectories across Days 1–10.
- **Downstream Impact Analytics:** Land-masked (`land_sea_mask >= 0.5`) Flood Risk, Heatwave Stress, and Moisture Surge grid-cell metrics synced to both presets and the interactive brush.

---

## 7. Repository Structure

```text
MonsoonFlow10/
├── serve.py                     # ROOT ENTRYPOINT: Offline FastAPI CPU inference server
├── requirements.txt             # Lean offline serving dependencies
├── project_brief.pdf            # 2-page technical submission brief
├── .gitignore                   # Excludes raw .nc data, node_modules, internal logs
│
├── checkpoints/                 # Committed Model A weights & precomputed arrays
│   ├── model.pt                 # exp4_msresunet_anchor weights (1.30 MB)
│   ├── stats.json               # Channel mean/std (2005–2018 training split)
│   ├── climatology.npy          # 366-day circularly smoothed seasonal baseline
│   └── static.npy               # 2x32x36 land_sea_mask and orography_m
│
├── models/                      # MultiScaleResUNet, SeasonalAnchorNet & checkpoint loader
├── inference/                   # Offline CPU predictor, schemas, and latency benchmark
├── data/                        # Chronological preprocessing, dataset, and static terrain
├── training/                    # Training & baseline evaluation pipelines
├── evaluation/                  # Scoring wrappers, physics audit, and certified scorecards
├── tests/                       # Pytest integrity, API contract & causality tests
├── dashboard/                   # Interactive 2D/3D React 19 + Three.js digital twin UI
└── guidelines/                  # DevForge Track 2 compliance documentation
```
