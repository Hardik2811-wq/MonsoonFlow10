## 1. Step 0: Lock In Your Agent's Rules (`CLAUDE.md`)

Before generating code, create a short `CLAUDE.md` file at the repository root so your AI coding agent (Claude Code / Antigravity) cannot silently violate hackathon rules or invent conflicting layouts. Keep it concise and focused on non-negotiable constraints and executable checks:

* **Entrypoint & Serving:** `serve.py` must live at the **repository root** (not inside `server/`), run via `python serve.py --port 8000` on `0.0.0.0`, and load `checkpoints/model.pt`, `stats.json`, and the `2 × 32 × 36` static tensor (`land_sea_mask`, `orography_m`) at import time.
* **Wire Format & Units:** `POST /simulate` receives and returns **physical units** (`t2m` in K, `tp` in mm/day, `u10`/`v10` in m/s, `msl` in Pa, `tcwv` in kg/m²) as finite JSON lists. Normalize inside `simulate()` using training-only `stats.json` and de-normalize before returning.
* **Model Inputs & Constraints:** Every forward step takes **10 channels** per grid cell (`6` dynamic + `2` static + `2` step-rotated `doy_sincos` planes). Enforce `tp >= 0`, `t2m` in `[233.15, 333.15]` K, and exact zero-shock parity (`atol < 1e-6`).
* **Anti-Leakage & Anti-Cheat:** Split `india_climate_train.nc` strictly chronologically (`2005–2018` train, `2019–2020` val, `2021` sealed test). Never import external weather APIs, never hardcode `YYYY-MM-DD` date strings in inference code, and never commit `.nc` files to git (keep repo `< 2 GB` with trained weights inside `checkpoints/`).
* **Definition of Done (Executable Gates):** No backend or ML task is complete until the agent runs and passes:
1. `python -m monsoon_r1.schema_validator http://localhost:8000`
2. `python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"`
3. Validation checks against `monsoon_r1.scoring` (`rmse_lat`, `csi`, `variance_ratio`, `causality_checks`, `null_consistent`).





---

## 2. Phase-by-Phase Execution Plan (Human + AI Agent)

Split the work into five bounded phases. Your job is to **set acceptance criteria, steer training runs on your RTX 4060, and inspect physical realism**; the agent's job is to **scaffold code, wire tests, run validators, and iterate on failing checks**.

| Phase | Core Deliverables | AI Agent Tasks | Human Steering & Verification Gate |
| --- | --- | --- | --- |
| **1. Bootstrap & Baselines** | Repo layout, `yug-monsoon-kit` setup, chronological dataset, `stats.json`, `climatology.npy`, baseline scores | Clone kit, extract `static` fields, generate 2005–2018 `stats.json` & `climatology.npy`, build PyTorch sliding-window dataset, write data-integrity tests (`test_data_integrity.py`) | Verify zero temporal overlap across splits and record exact Day 1, 4, and 8 `rmse_lat` and `10.0 mm/day` CSI floors for Persistence and Climatology on 2019–2020 |
| **2. MF10 Core & Curriculum Training** | `models/mf10.py`, `models/physics_losses.py`, `training/train.py`, rollout evaluator | Implement 10-channel Residual CNN Encoder → 2-layer ConvLSTM (64 ch) → Residual Decoder (`ΔX`), step-wise `doy_sincos` rotation matrix, composite loss (lat/std-weighted forecast + heavy-rain `>= 10 mm/day` + Day-8 variance floor `>= 0.55`), and 1 → 3 → 5 → 10 day curriculum script | Run training on RTX 4060; inspect TensorBoard/plots after each curriculum stage to confirm Day-4 beats persistence, `tp` doesn't blur into drizzle, and Day-8 `variance_ratio` stays in `[0.40, 2.50]` |
| **3. Shock Engine & Root `serve.py**` | Perturbation validation, root `serve.py`, CPU latency benchmark | Wire physical delta `P` into `init_states[-1]` (`X'_t = X_t + P`), run `monsoon_r1.scoring.causality_checks` for `warm_ocean`, `monsoon_stall`, and `zero_shock`, build root `serve.py` (`/health`, `/simulate`), benchmark 10-step CPU inference (4 threads) | Confirm all 4 causality checks pass (localized Day-1 `>= 60%`, Day 4–8 outward spread `> 1.5×`, no teleportation, max shift `<= 5σ`, null diff `<= 1e-3σ`) and CPU latency is well under `5.0 s` (target `< 0.5 s`) |
| **4. Multiverse Dashboard & Impact Module** | `dashboard/` (React + TypeScript + Vite + Tailwind + Leaflet + Recharts) | Scaffold frontend completely separate from `serve.py`, wire live `POST /simulate` calls, build India map layers, Day 0–10 scrubber, Baseline vs. Shock anomaly view (`ΔX_t`), Gaussian-smoothed custom shock controls, realism warnings, and flood/crop-stress impact cards | Test live end-to-end against `serve.py` (no mocked/pre-rendered data); verify a judge can apply an on-the-spot shock during Desk Review and see crisp anomaly maps and impact shifts |
| **5. Final Freeze & 2-Page Brief** | Sealed 2021 test report, passing repo scan, `project_brief.pdf` | Run `schema_validator` and `repo_scan.scan('.')`, run one-shot evaluation on sealed 2021 holdout, verify `checkpoints/model.pt` is committed and `.nc` is gitignored, generate clean diagnostic plots for the brief | Review static scan for 0 critical flags, verify repo size `< 2 GB`, write and polish the 2-page PDF Brief (Page 1: MF10 architecture, curriculum, losses, limits; Page 2: Multiverse UI & flood/crop impact case study)

 |

---

## 3. How to Run Your Agent Effectively (Workflow & Prompts)

Use a strict **Plan → Implement → Verify** loop for every task so the agent never spirals into massive unverified rewrites:

1. **Keep sessions scoped by directory:** Run one agent session for **ML + Backend** (`data/`, `models/`, `training/`, `serve.py`) and—once the `/simulate` JSON contract is frozen—spin up a separate parallel session or subagent exclusively for **`dashboard/`**. Explicitly tell the frontend agent not to touch Python files.
2. **Use `monsoon_r1` as your agent's automated test harness:** Instead of asking the agent to "make sure the model works," point it directly at `monsoon_r1.scoring` and `monsoon_r1.schema_validator`. Agents iterate dramatically faster when they can read exact pass/fail numbers from a script.
3. **Prune the 7-instrument test suite for hackathon speed:** Do not ask the agent to build the full `evaluation/` package from your research notes before training a model. Build only three lean scripts first: (a) `test_data_integrity.py`, (b) `eval_scoring.py` wrapping `monsoon_r1.scoring` for Pool A (2019–2020), and (c) the official `schema_validator` + `repo_scan` commands.



**Starter Prompt for Phase 1 (Copy-Paste to Your Agent):**

> Read `CLAUDE.md` and inspect `resources/yug-monsoon-kit/monsoon_r1/contract.py` and `scoring.py`. First, give me a step-by-step plan (no code yet) to: (1) extract `static` fields and compute training-only `stats.json` and `climatology.npy` strictly from 2005–2018 of `india_climate_train.nc`, (2) build a chronological PyTorch Dataset (`2005–2018` train, `2019–2020` val, `2021` sealed test), and (3) compute Day 1, 4, and 8 `rmse_lat` and `10.0 mm/day` CSI baselines for Persistence and Climatology on 2019–2020. Wait for my approval before writing files.

**Starter Prompt for Phase 2 (Copy-Paste After Phase 1 Passes):**

> Implement `models/mf10.py` (10-channel input: 6 dynamic + 2 static + 2 step-rotated `doy_sincos`, Residual CNN Encoder → 2-layer 64-ch ConvLSTM → Residual Decoder predicting `ΔX`, with `tp >= 0` enforcement) and `models/physics_losses.py` (latitude/std-weighted forecast loss, `>= 10.0 mm/day` heavy-rain loss, and Day-8 spatial variance floor `>= 0.55`). Write a fast unit test verifying output shape `(B, 10, 6, 32, 36)`, zero-shock exact parity (`atol=1e-6`), and finite gradients on dummy tensors before running training on GPU.

---

## 4. High-Risk Agent Traps to Watch For

* **Hallucinated `serve.py` paths or normalization leaks:** Agents often normalize inside `serve.py` and forget to de-normalize `rollout` back to Kelvin/Pa/mm-per-day, or place `serve.py` in `server/`. Catch this immediately with `schema_validator`.
* **Tripping `repo_scan.py` silently:** Agents love writing helper utilities with `datetime` or example test dates (`"2023-07-15"`). If `serve.py` transitively imports any file with `>= 5` hardcoded date literals or external URLs, you lose anti-cheat points. Keep training/test utilities strictly unimported by `serve.py`.
* **Over-smoothing under MSE:** If the agent defaults to plain MSE during Phase 2, your model will look decent on Day-1 RMSE but fail the Day-8 Anti-Blur check (`variance_ratio < 0.40`) and Heavy-Rain CSI. Check `variance_ratio` and `csi` on 2019–2020 right after the 3-day curriculum stage.
* **Uncontrolled scope creep:** Agents will happily spend an hour building complex UI animations or custom Dockerfiles you don't need. Remember Track 2 does not use a custom Dockerfile—it runs `python serve.py --port 8000` inside the organizer's offline CPU container.



