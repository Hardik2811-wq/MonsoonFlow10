# MonsoonFlow-10 — Independent Model Audit & Scorecard

**Audit date:** 2026-11-27 · **Auditor:** Claude audit session (read-only on model, training, inference and evaluation code)
**Artefact audited:** `checkpoints/model.pt`, 11,651,378 bytes (11.11 MiB), md5 `58e134fb69fe33605a577d7d32e88adf`
**Inputs:** `evaluation/eval_scorecard_pool_a.json`, `evaluation/eval_scorecard_2021_holdout.json`, `models/`, `inference/`, `serve.py`, `monsoon_r1` kit.

Every number below was re-measured during this audit or reproduced from the stored scorecards. Scripts and raw outputs are in [`modelScore/repro/`](repro/) and can be rerun from the repo root (`python modelScore/repro/audit_model.py`, `python modelScore/repro/ablate.py`). No weights, training code, inference code or evaluation files were modified.

---

## 0. Verdict

| | |
|---|---|
| **Scorecards reproduce exactly** | Re-running `evaluation/eval_scoring.evaluate_checkpoint_model` on the same checkpoint gives **65.270** (Pool A) and **68.263** (2021), identical to the stored JSON. |
| **Hard constraints** | All pass: admission gate (validity + beats Day-4 persistence) on both sets; kit `schema_validator` against live `serve.py` "all checks passed"; `pytest tests/` 11/11; outputs finite, `tp ≥ 0`, `t2m` in bounds; zero shock exactly equal to baseline (max \|Δ\| = 0.0). |
| **Biggest point gap** | RMSE skill: SS ≈ **+0.03** against the full-marks reference of 0.12, so only ~7 of 30 points. This is the main lever. |
| **Main headline caveats** | (1) `monsoon_stall` fails `propagates` only because the evaluator's shock window is **3 January** (north-east monsoon). It passes in July–August windows. (2) The evaluator scores only the **first 50 windows** (Jan–Jul); the full split is worth about **+5 skill points** on Pool A. (3) `repo_scan_pts` and `brief_pts` in the scorecards are **hard-coded 15/15 defaults**, not measurements. I verified the repo scan independently (15/15 on the clean tracked tree); the brief score remains an assumption. |
| **Where the skill comes from** | Ablation: the climatology-anchored physics prior supplies essentially **all of the RMSE skill**. The learned neural increment supplies essentially **all of the heavy-rain CSI skill**. |

---

## 1. Executive scorecard

Official protocol = `evaluation/eval_scoring.py` (wraps `monsoon_r1.scoring`): 50 windows (stride 4, from 1 January of the split), one shock window (window 0).

| Component | Max | Pool A 2019–2020 | Sealed 2021 holdout | Audit status |
|---|---:|---:|---:|---|
| **Admission gate** | pass/fail | **PASS** (0 violations) | **PASS** (0 violations) | Reproduced |
| ↳ Day-4 mean lat-RMSE, MF10 vs persistence | — | 47.34 vs 61.18 | 51.63 vs 65.12 | Reproduced |
| **Forecast skill** | 45 | **12.14** | **15.14** | Reproduced |
| ↳ RMSE skill (SS vs best of persistence/climatology; full marks at 0.12) | 30 | 6.75 (SS +0.0270) | 7.05 (SS +0.0282) | Reproduced |
| ↳ Heavy-rain CSI ≥ 10 mm/d (full marks at CSI-SS 0.10) | 15 | 5.40 (CSI-SS +0.0360) | 8.09 (CSI-SS +0.0539) | Reproduced |
| **Dynamics & multiverse** | 25 | **23.125** | **23.125** | Reproduced |
| ↳ Anti-blur (Day-8 variance ratio in [0.40, 2.50]) | 10 | 10.0 | 10.0 | Reproduced; all channels in band |
| ↳ Causality (2 shocks × 4 checks, null halves) | 15 | 13.125 | 13.125 | Reproduced; `monsoon_stall` propagates ✗ (see §4) |
| **Repo scan** | 15 | 15.0 | 15.0 | **Hard-coded default** in `eval_scoring.py` (`repo_scan_points=15.0`). Independently verified: clean tracked tree scores 15/15 with 0 flags (§5) |
| **Brief** | 15 | 15.0 | 15.0 | **Hard-coded default** (`brief_points=15.0`); manual-review item, not measured |
| **Total** | **100** | **65.27** | **68.26** | Reproduced exactly |

**Sensitivity to evaluation protocol.** These figures are measured; the official grader's exact window protocol is unknown.

| Scenario | Pool A | 2021 |
|---|---:|---:|
| Official (first 50 windows, January shock window) | 65.27 | 68.26 |
| All windows (180 Pool A / 89 in 2021), January shock window | 70.43 | 70.72 |
| All windows + a July/August shock window (stall propagates → causality 15.0) | 72.31 | 72.59 |
| Same, but brief scored below 15 | subtract the shortfall | subtract the shortfall |

---

## 2. Per-channel and lead-day analysis (Days 1, 4, 8)

Skill = 1 − RMSE_MF10 / min(RMSE_persistence, RMSE_climatology), latitude-weighted, per channel. Positive means MF10 beats the stronger reference.

#### Pool A 2019–2020 (official 50-window sample)

| Channel | Lead | MF10 RMSE | Persistence | Climatology | Best ref | Skill vs best ref |
|---|---|---:|---:|---:|---|---:|
| t2m (K) | D1 | 1.130 | 1.117 | 1.837 | Persist | -0.011 |
| t2m (K) | D4 | 1.763 | 2.054 | 1.824 | Clim | +0.034 |
| t2m (K) | D8 | 1.796 | 2.263 | 1.812 | Clim | +0.009 |
| tp (mm/d) | D1 | 4.646 | 5.386 | 5.863 | Persist | +0.137 |
| tp (mm/d) | D4 | 6.072 | 8.164 | 6.072 | Clim | +0.000 |
| tp (mm/d) | D8 | 6.032 | 8.609 | 5.997 | Clim | -0.006 |
| u10 (m/s) | D1 | 1.070 | 1.144 | 1.428 | Persist | +0.065 |
| u10 (m/s) | D4 | 1.387 | 1.811 | 1.397 | Clim | +0.008 |
| u10 (m/s) | D8 | 1.382 | 1.963 | 1.392 | Clim | +0.007 |
| v10 (m/s) | D1 | 1.019 | 1.095 | 1.388 | Persist | +0.069 |
| v10 (m/s) | D4 | 1.388 | 1.797 | 1.386 | Clim | -0.002 |
| v10 (m/s) | D8 | 1.371 | 1.940 | 1.357 | Clim | -0.010 |
| msl (Pa) | D1 | 195.7 | 213.2 | 272.0 | Persist | +0.082 |
| msl (Pa) | D4 | 267.8 | 346.1 | 276.9 | Clim | +0.033 |
| msl (Pa) | D8 | 268.2 | 340.0 | 275.8 | Clim | +0.028 |
| tcwv (kg/m²) | D1 | 3.314 | 3.584 | 5.588 | Persist | +0.075 |
| tcwv (kg/m²) | D4 | 5.691 | 7.121 | 5.647 | Clim | -0.008 |
| tcwv (kg/m²) | D8 | 5.680 | 7.598 | 5.542 | Clim | -0.025 |

#### Sealed 2021 holdout (official 50-window sample)

| Channel | Lead | MF10 RMSE | Persistence | Climatology | Best ref | Skill vs best ref |
|---|---|---:|---:|---:|---|---:|
| t2m (K) | D1 | 1.104 | 1.098 | 1.826 | Persist | -0.006 |
| t2m (K) | D4 | 1.778 | 2.084 | 1.869 | Clim | +0.049 |
| t2m (K) | D8 | 1.800 | 2.345 | 1.843 | Clim | +0.023 |
| tp (mm/d) | D1 | 4.972 | 5.538 | 6.202 | Persist | +0.102 |
| tp (mm/d) | D4 | 6.473 | 8.367 | 6.422 | Clim | -0.008 |
| tp (mm/d) | D8 | 6.581 | 8.854 | 6.523 | Clim | -0.009 |
| u10 (m/s) | D1 | 1.059 | 1.088 | 1.591 | Persist | +0.028 |
| u10 (m/s) | D4 | 1.602 | 2.061 | 1.602 | Clim | +0.000 |
| u10 (m/s) | D8 | 1.608 | 2.315 | 1.612 | Clim | +0.002 |
| v10 (m/s) | D1 | 0.993 | 1.042 | 1.403 | Persist | +0.046 |
| v10 (m/s) | D4 | 1.373 | 1.758 | 1.422 | Clim | +0.035 |
| v10 (m/s) | D8 | 1.415 | 1.891 | 1.421 | Clim | +0.004 |
| msl (Pa) | D1 | 197.5 | 215.4 | 277.2 | Persist | +0.083 |
| msl (Pa) | D4 | 292.7 | 369.4 | 301.3 | Clim | +0.029 |
| msl (Pa) | D8 | 297.7 | 418.2 | 298.5 | Clim | +0.003 |
| tcwv (kg/m²) | D1 | 3.319 | 3.575 | 6.304 | Persist | +0.072 |
| tcwv (kg/m²) | D4 | 5.787 | 7.082 | 6.256 | Clim | +0.075 |
| tcwv (kg/m²) | D8 | 6.155 | 8.084 | 6.028 | Clim | -0.021 |

### Where MF10 gains the most

- **Day 1, every channel except t2m.** MF10 beats persistence on 5 of 6 channels:
  - **tp +0.137 / +0.102** (Pool A / 2021), the largest single gain;
  - msl +0.082 / +0.083;
  - tcwv +0.075 / +0.072;
  - u10 and v10 +0.03 to +0.07.

  The anomaly formulation keeps today's anomaly but relaxes it towards the seasonal climatology and advects it with the wind. That beats raw persistence for fast-decorrelating fields (rain, wind, pressure systems).
- **Day 4: t2m, msl and tcwv.** t2m +0.034 / +0.049 and msl +0.033 / +0.029. In 2021, tcwv +0.075. The thermodynamic damping r ≈ 0.965 keeps 0.965⁴ ≈ 87 % of the anomaly at Day 4, while climatology carries none of it.
- **Monsoon season (JJAS) is the strongest regime for thermodynamics.** Pool A JJAS Day 4: t2m +0.099, tcwv +0.114. 2021 JJAS Day 4: tcwv +0.136. Full-split numbers are in `repro/audit_results.json` (`full.*.JJAS.ss_by_day`).

### Where climatology stays competitive

- **From Day 4 onward the stronger reference is climatology for every channel** (at Days 4–8 persistence RMSE is 13–44 % higher than climatology's). MF10 is then within ±0.05 of climatology, by design. The learned dynamic-channel damping is r ≈ 0.797 for tp, u10, v10 and msl, so their anomalies keep 0.80⁴ ≈ 41 % at Day 4 and 0.80⁸ ≈ 17 % at Day 8. The forecast therefore converges to climatology.
- **Small negatives against climatology:**
  - tp at Days 4 and 8 (−0.006 to −0.009);
  - tcwv at Day 8 (−0.025 / −0.021);
  - v10 at Day 8 in Pool A (−0.010).

  These are below the noise of a 50-window sample, but they show there is no residual skill left at Day 8 to collect.
- **t2m at Day 1 is the one channel where persistence wins** (−0.011 / −0.006). Near-surface temperature is highly persistent day to day, and the Day-1 pull towards climatology plus damping costs a little. A lead-dependent blend is the fix (§6).

### Unit note

The admission gate compares the **mean of six per-channel RMSEs in physical units**. msl (~200–350 Pa) dominates that mean (47.3 of it is mostly msl/6 ≈ 45). The gate is therefore effectively an msl test. MF10 passes it with a wide margin: 267.8 vs 346.1 Pa (Pool A, Day 4).

---

## 3. Heavy-rain CSI and anti-blur audit

#### Heavy-rain CSI at 10 mm/day

| Set | Lead | MF10 | Persistence | Climatology | CSI skill vs best ref |
|---|---|---:|---:|---:|---:|
| Pool A | D1 | 0.465 | 0.425 | 0.306 | +0.069 |
| Pool A | D4 | 0.332 | 0.194 | 0.310 | +0.031 |
| Pool A | D8 | 0.346 | 0.179 | 0.340 | +0.008 |
| 2021 | D1 | 0.431 | 0.399 | 0.272 | +0.053 |
| 2021 | D4 | 0.317 | 0.216 | 0.265 | +0.071 |
| 2021 | D8 | 0.305 | 0.188 | 0.277 | +0.038 |

#### Day-8 spatial variance ratio (pred / truth); pass band [0.40, 2.50]

| Set | t2m | tp | u10 | v10 | msl | tcwv | Anti-blur pts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pool A (official 50) | 0.949 | 0.722 | 0.821 | 0.715 | 0.839 | 0.930 | 10.0 / 10 |
| 2021 (official 50) | 1.016 | 0.520 | 0.739 | 0.896 | 0.897 | 0.901 | 10.0 / 10 |
| Pool A all 180 windows | 0.991 | 0.722 | 0.737 | 0.708 | 0.914 | 0.913 | 10.0 / 10 |
| Pool A JJAS (61) | 1.005 | 0.497 | 0.901 | 0.834 | 0.972 | 0.933 | 10.0 / 10 |
| 2021 all 89 windows | 1.007 | 0.597 | 0.719 | 0.780 | 0.900 | 0.902 | 10.0 / 10 |
| 2021 JJAS (30) | 1.005 | 0.531 | 0.876 | 0.832 | 0.960 | 0.937 | 10.0 / 10 |

**CSI.** MF10 beats both references at every lead on both sets. The CSI skill is held back at Day 8 on the official Pool A sample (+0.008), where climatology's CSI (0.340) nearly equals MF10's (0.346). That sample is the first 50 windows (January to mid-July), which contain few ≥ 10 mm/d events outside the monsoon onset.

**Window sensitivity.** Over all windows, CSI-SS is +0.067 (Pool A) and +0.065 (2021), worth about 10 of 15 points instead of 5.4 and 8.1. The 50-window sample is not seasonally representative.

#### Window-sample sensitivity (forecast-skill points out of 45)

| Set | Windows | t0 range | RMSE SS | RMSE pts /30 | CSI SS | CSI pts /15 | Skill pts /45 | Day-4 MF10 vs Persist |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Pool A official first 50 | 50 | 20190103–20190718 | +0.0270 | 6.75 | +0.0360 | 5.40 | 12.14 | 47.34 vs 61.18 |
| Pool A all windows | 180 | 20190103–20201219 | +0.0291 | 7.27 | +0.0669 | 10.04 | 17.31 | 47.43 vs 59.44 |
| Pool A JJAS only | 61 | 20190604–20200930 | +0.0355 | 8.89 | +0.0598 | 8.97 | 17.85 | 43.17 vs 53.72 |
| Pool A non-JJAS | 119 | 20190103–20201219 | +0.0280 | 6.99 | +0.0765 | 11.47 | 18.46 | 49.35 vs 62.03 |
| 2021 official first 50 | 50 | 20210103–20210718 | +0.0282 | 7.05 | +0.0539 | 8.09 | 15.14 | 51.63 vs 65.12 |
| 2021 all windows | 89 | 20210103–20211221 | +0.0314 | 7.85 | +0.0649 | 9.74 | 17.59 | 49.05 vs 62.43 |
| 2021 JJAS only | 30 | 20210604–20210928 | +0.0285 | 7.13 | +0.0586 | 8.79 | 15.92 | 41.65 vs 54.11 |
| 2021 non-JJAS | 59 | 20210103–20211221 | +0.0338 | 8.44 | +0.0760 | 11.39 | 19.84 | 52.35 vs 66.20 |

**Anti-blur.** Every channel is inside [0.40, 2.50] in every slice audited, so all slices score 10/10. The **tightest channel is tp**:
- 0.497 in the Pool A JJAS windows;
- 0.520 on the official 2021 sample.

That leaves a margin of about 0.1 above the 0.40 floor. Any change that smooths rain further (more damping, more diffusion, MSE-only training) risks this item first. t2m sits at 0.95–1.02, which is ideal.

**Neural increment ablation.** Zeroing the decoder output layer leaves only the physics prior (climatology + damped anomaly + transport). Measured over all windows:

| Set | Variant | RMSE SS | CSI-SS | Skill pts /45 | Day-4 lat-RMSE |
|---|---|---:|---:|---:|---:|
| Pool A | MF10 (full) | +0.0291 | +0.0669 | 17.31 | 47.43 |
| Pool A | prior only (head zeroed) | +0.0286 | +0.0127 | 9.06 | 47.33 |
| 2021 | MF10 (full) | +0.0314 | +0.0649 | 17.59 | 49.05 |
| 2021 | prior only (head zeroed) | +0.0303 | +0.0032 | 8.05 | 49.15 |

The trained network is worth about **+8 to +9.5 forecast-skill points, almost entirely through heavy-rain CSI**. It adds essentially nothing to RMSE skill, which comes from the climatology anchoring.

---

## 4. Shock causality and physics audit

Checks follow `monsoon_r1.scoring.causality_checks`:

- **finite** — no NaN or Inf in the response.
- **nonzero** — the shock moves some channel by at least 1e-3 standard deviations.
- **stable** — the response stays below the kit's stability ceiling.
- **localized** — on Day 1, at least 60 % of the response energy lies within 3 cells of the box.
- **propagates** — the mean energy outside that 3-cell neighbourhood over Days 4–8 exceeds 1.5× its Day-1 value, and exceeds 2 % of the peak total energy.
- **no_teleport** — less than 5 % of the energy lies beyond 3 + 8·(k+1) cells from the box.

I swept the official perturbations over **60 start windows in Pool A and 30 in 2021** (every 3rd window, roughly every 12 days).

### `zero_shock` (null consistency)

- Max |shock − baseline| / std = **0.0 exactly** in all 90 windows; the tolerance is 1e-3.
- `serve.py` and `inference/predictor.py` bypass the addition entirely when the perturbation is `null` or all zeros (\|P\| ≤ 1e-12), so the result is bit-identical. The kit `schema_validator` "zeros" case passes against the live server.

### `warm_ocean` (+3 K t2m and +15 % tcwv over 8–20°N, 82–94°E)

| Check | Pool A (60 windows) | 2021 (30 windows) | Official window |
|---|---:|---:|---|
| localized (Day-1 near fraction) | 60/60 | 30/30 | 0.9988 / 0.9983 |
| propagates | **59/60** (single miss in October) | **30/30** | ✓ |
| no_teleport (far fraction) | 60/60 | 30/30 | 0.0 |
| stable | 60/60 | 30/30 | ✓ |

The perturbed channels are t2m and tcwv, whose anomaly damping is r ≈ 0.965, so the shock persists. The bounded transport (≤ 8 cells/day, CFL ≤ 0.95, diffusivity κ ≈ 0.19) spreads it outward from an interior box. It propagates in essentially every season.

### `monsoon_stall` (−40 % of u10 and v10 over 8–20°N, 65–75°E)

`localized`, `no_teleport` and `stable` pass in **90/90** windows. `propagates` passes in only **19/60 (Pool A)** and **9/30 (2021)** windows, and the pattern is purely seasonal:

| Month of t0 | Pool A pass | 2021 pass | Mean u10 in box, Pool A (m/s) | Mean u10 in box, 2021 (m/s) |
|---|---:|---:|---:|---:|
| Jan | 0/5 | 0/3 | −2.79 | −1.58 |
| Feb | 0/5 | 0/2 | −0.46 | −3.09 |
| Mar | 0/5 | 0/3 | +0.65 | −0.23 |
| Apr | 0/5 | 0/2 | +1.15 | +1.45 |
| May | 2/5 | 1/3 | +3.08 | +5.17 |
| Jun | 3/5 | 2/2 | +5.91 | +7.75 |
| **Jul** | **5/5** | **3/3** | +7.25 | +8.79 |
| **Aug** | **6/6** | 2/3 | +7.09 | +6.60 |
| Sep | 3/5 | 1/2 | +4.66 | +6.68 |
| Oct | 0/5 | 0/3 | +0.90 | +0.92 |
| Nov | 0/5 | 0/2 | −2.57 | −0.17 |
| Dec | 0/4 | 0/2 | −3.01 | −1.75 |

**Why the official scorecards record `propagates=False`.** The evaluator's shock window is window 0 = **t0 = 3 January** in both sets. On that day the box winds are north-easterly: u, v = −3.44, −3.36 m/s (Pool A) and −1.47, −4.73 m/s (2021).

**Mechanism:**

1. **Geometry.** The box's western edge (65°E) lies outside the first grid column (65.5°E), so the box's west column *is* the domain's open western boundary. Its southern row (8.5°N) is 2 cells from the southern boundary (6.5°N). Dilated by 3 cells, the "near" neighbourhood covers everything to the west and south up to the domain edge. Energy can only count as "outward" if it travels roughly **north of ~23°N or east of ~78°E**.
2. **Winter (NE monsoon, Oct–Apr).** The transport advects the wind anomaly with the background flow, towards the south-west and out through the open 65.5°E and 6.5°N boundaries. The model has no inflow or halo beyond the 32×36 domain, so that energy leaves the system. What stays inside decays at r ≈ 0.80/day (wind channels), so Days 4–8 outward energy never reaches 1.5× the Day-1 value. The shock is also proportional to the local wind (−40 % of a weak 2–4 m/s wind), so it starts small.
3. **Summer (SW monsoon, Jun–Aug).** The Somali-jet westerlies (u ≈ +6 to +9 m/s) carry the anomaly **east over peninsular India**, out of the 3-cell neighbourhood within 1–3 days. The check passes in every July window (8/8) and 8 of 9 August windows. This matches the dashboard audit: both graded shocks passed all four checks on the mid-July 2019 initial state.

This is physically reasonable behaviour: in winter the real atmosphere also carries an Arabian Sea wind anomaly offshore. But it costs **1.875 causality points** whenever the grader's shock window falls between October and April.

**Points:**

| Shock window | Causality points |
|---|---:|
| January (official sample) | 13.125 / 15 |
| June to September | 15.0 / 15 |

Not verified here: whether the organisers' grader uses a fixed window, a random one, or several. That is worth asking (§6).

---

## 5. Offline CPU latency and anti-cheat audit

### Latency (production path `inference.predictor.simulate`, `torch.set_num_threads(4)`)

25 consecutive 10-step rollouts on the development machine:

| Metric | Value |
|---|---:|
| First call (cold) | 70.9 ms |
| Median (warm) | **57.7 ms** |
| p95 | 65.8 ms |
| Max | 66.2 ms |
| Limit / target | 5,000 ms / 500 ms → ~**75×** headroom under the limit |

The kit `schema_validator` against the live server (`http://127.0.0.1:8000`, including JSON (de)serialisation over HTTP) took 0.10–0.14 s per request in all 6 cases (canonical JSON and base64 × none / warm_ocean / zeros). Output check: shape `(10, 6, 32, 36)`, all finite, `tp` min 0.0, `t2m` 246.5–301.0 K on a January window.

*Not verified:* latency inside the official container (`--network none`, 4 CPU cores, 6 GB RAM). Per-core speed there may differ from this machine, but a ~75× margin leaves ample room.

### Model size

| Item | Size |
|---|---:|
| `checkpoints/model.pt` | 11,651,378 B = **11.11 MiB** (11.65 MB decimal) |
| ↳ `clim_norm` buffer (365 × 6 × 32 × 36 fp32 climatology table) | 10.09 MB (**87 %** of the file) |
| ↳ learned parameters: **387,736** (encoder, 2-layer ConvLSTM, decoder, transport, damping) | ~1.55 MB |

### Repository size (claimed 0.343 GB; measured now)

| Measurement | Size |
|---|---:|
| Tracked files at HEAD (`git archive HEAD`) | **0.046 GB** |
| `.git` object store | 0.035 GB |
| Whole working directory | 0.392 GB, of which `dashboard/node_modules` is 0.192 GB and `data/india_climate_train.nc` 0.123 GB (both git-ignored) |

All are far below the 2 GB limit. The earlier 0.343 GB figure was presumably a working-tree measurement at another moment; it doesn't match any measurement now, but the conclusion is unchanged.

### `repo_scan` (kit `monsoon_r1.repo_scan.scan`)

| Scan target | Points | Anti-cheat | Health | Flags |
|---|---:|---:|---:|---|
| **Clean tracked tree** (`git archive HEAD`, what a grader clones) | **15.0** | 8.0 | 7.0 | **0** |
| Local working tree | 5.0 | 0.0 | 5.0 | 12 critical + 1 warn |

- **Clean tree:** 17 inference files detected (`serve.py`, `inference/`, `models/`, `data/`). `server_separate_from_ui` is true.
- **Working-tree flags:** all 12 critical flags are the kit's own blocklist inside the git-ignored `resources/yug-monsoon-kit/monsoon_r1/repo_scan.py`. The warning is the git-ignored `.nc` file. Because of that kit folder, the local scan also sees the server as not separate from the UI.
- **Discrepancy to fix:** `evaluation/repo_scan_report.json` stores the 5/15 working-tree result, while both scorecards claim 15/15 from the default argument. The 15 is correct for a submission, but for the wrong reason. Agent-QA should regenerate the report from a clean export, or exclude `resources/`.

### Leakage and anti-cheat

- **Chronological split by integer year:** train 2005–2018 (5,113 days), val 2019–2020 (731), test 2021 (365). The climatology buffer comes from the training split.
- **Holdout governance:** `model.pt` is byte-identical (same md5) to `checkpoints/model_phase4_backup.pt`, written at 20:21. That is **before** the 2021 holdout evaluation at 23:35. The only later edits to `models/` (00:03) are comment-only (diff checked). **No evidence of tuning on the holdout.**
- **Leftover candidate:** `model_phase6_candidate.pt` (post-holdout) is still tracked at HEAD, although deleted in the working tree. It is not the deployed model.
- **No external weather sources and no tracked `.nc`/`.grib`/`.zarr` files** (`git ls-files`).
- **Date literals:** the scan passes on inference code. `evaluation/eval_scoring.py` (not inference) uses `dt.date(2000, …)` for climatology indexing. The audit's own repro scripts use a month-offset table instead, so they add no date literals.

---

## 6. Honest limitations and improvement paths

1. **RMSE skill is the ceiling (≈ 7/30 points).**
   - **Problem:** SS ≈ 0.03 against the full-marks reference of 0.12. The ablation shows the trained network contributes almost nothing to RMSE skill (+0.0005 Pool A, +0.0011 in 2021).
   - **Path:** the machine has an RTX 5070, but the installed torch is `2.15.0+cpu`. A CUDA build and a longer **multi-lead rollout loss** (Days 1–10, weighted toward Days 4 and 8, which are scored) is the single largest expected gain.
   - **Also:** weight each channel's loss by 1 / RMSE_clim, so that msl in Pa does not dominate training the way it dominates the gate.
2. **Day-1 t2m loses to persistence.** Use a lead-dependent anomaly retention, so the Day-1 forecast stays close to persistence and relaxes to climatology only at longer leads. For example, condition r on lead, or let the increment see the persistence error.
3. **`monsoon_stall` in non-monsoon windows.**
   - **Root cause:** an open western/southern boundary with no inflow, plus north-easterly advection.
   - **Legitimate paths:**
     - **(a)** Couple the wind anomaly to convergence, moisture and pressure responses, so a stall leaves an interior imprint (tcwv, tp, msl) that spreads even when the wind anomaly itself is carried offshore.
     - **(b)** Add a sponge or halo boundary that holds outgoing anomaly near the edge physically, rather than deleting it.
   - **To avoid:** anything that pushes energy north or east regardless of the wind just to satisfy the check. That would be gaming the metric, and judges reviewing causality will look for it.
   - **Also:** ask the organisers which shock window(s) the grader uses.
4. **Evaluation protocol bias (owner: Agent-QA).**
   - **Problem:** the first-50 sample covers January to mid-July only and under-states skill by about 5 points on Pool A. The single January shock window determines 1.875 points.
   - **Fix:** report all windows or a season-stratified sample, and a multi-window causality pass rate (as in §4).
   - **Also verify:** the evaluator applies shocks via `model(..., perturbation=)`, while `serve.py` adds them to `init_states[-1]` with physical clipping. This audit did not check that the two paths are numerically equivalent; a one-line test comparing them would close that gap.
5. **Precipitation sharpness margin.** The tp Day-8 variance ratio is ≈ 0.50 in monsoon windows. More damping, diffusion or MSE weighting on rain could push it under 0.40 and cost anti-blur points. Keep the L1 / heavy-rain loss terms.
6. **Unmeasured items:**
   - the brief's 15 points (manual);
   - latency inside the official 4-core, 6 GB offline container;
   - the organisers' exact window protocol.
7. **Housekeeping:**
   - Regenerate `evaluation/repo_scan_report.json` from a clean export.
   - Untrack `checkpoints/model_phase4_backup.pt` (identical to `model.pt`) and `model_phase6_candidate.pt`: about 23 MB of redundant binaries.
   - `/health` reports `"ready"` even when weights fail to load (noted in the dashboard audit).

---

## Appendix — method and reproduction

| Step | How |
|---|---|
| Scorecard reproduction | `evaluate_checkpoint_model(model.pt, 50, "val" \| "test")` → 65.270 / 68.263 |
| Full split and season slices | Same model and dataset (`step_stride=4`), all windows, scored with `evaluate_rollouts`. Climatology rows indexed by month/day (leap-indexed table). |
| Causality sweep | Every 3rd window. The kit's `causality_checks` with the evaluator's exact perturbations (warm: +3 K, +15 % tcwv; stall: −40 % u10/v10 of the t0 state). |
| Ablation | `decoder.out_conv[-1]` weight and bias zeroed, otherwise identical. |
| Latency | `inference.predictor.simulate` after `registry.load_artifacts()`, 25 calls, 4 threads. |
| Repo scan | `monsoon_r1.repo_scan.scan` on `git archive HEAD` (clean) and on the working tree. |
| Live contract | `python -m monsoon_r1.schema_validator http://127.0.0.1:8000` → all checks passed. `pytest tests` → 11 passed. |

Raw outputs: [`repro/audit_results.json`](repro/audit_results.json), [`repro/ablation.json`](repro/ablation.json).
