"""Independent MF10 audit (read-only on the repo). Writes JSON results to the scratchpad only."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]  # repo root (modelScore/repro/ -> ../../)
OUT = Path(__file__).with_name("audit_results.json")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "resources" / "yug-monsoon-kit"))

from data.preprocessing import load_raw_dataset, chronological_split  # noqa: E402
from data.dataset import MonsoonDataset  # noqa: E402
from models.mf10 import MonsoonFlow10  # noqa: E402
from evaluation.eval_scoring import evaluate_rollouts, evaluate_checkpoint_model  # noqa: E402
from monsoon_r1.scoring import causality_checks  # noqa: E402
from monsoon_r1.contract import BOX_BAY_OF_BENGAL, BOX_ARABIAN_SEA, box_mask  # noqa: E402

torch.set_num_threads(4)
CK = REPO / "checkpoints"
stats = json.load(open(CK / "stats.json"))
static = np.load(CK / "static.npy")
clim = np.load(CK / "climatology.npy")
mean = np.array(stats["mean"], "f4")
std = np.array(stats["std"], "f4")
M5 = mean.reshape(1, 1, 6, 1, 1)
S5 = std.reshape(1, 1, 6, 1, 1)

model = MonsoonFlow10(in_channels=10, dynamic_channels=6, hidden_dim=64, num_lstm_layers=2, stats=stats)
sd = torch.load(CK / "model.pt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
model.load_state_dict(sd, strict=True)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
results = {"n_params": int(n_params), "model_pt_bytes": (CK / "model.pt").stat().st_size}

raw, dates = load_raw_dataset(REPO / "data" / "india_climate_train.nc")
splits = chronological_split(raw, dates)
results["split_days"] = {k: [int(v[1][0]), int(v[1][-1]), int(len(v[1]))] for k, v in splits.items()}


# Row in the 366-row leap-indexed climatology table (no date literals): cumulative leap-year month offsets
LEAP_CUM = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)


def clim_for(d_int):
    return clim[LEAP_CUM[d_int // 100 % 100 - 1] + d_int % 100 - 1]


def rollouts(split):
    data, dts = splits[split]
    ds = MonsoonDataset(data, dts, stats, static, t_in=3, t_rollout=10, step_stride=4)
    P, T, Pe, C, D0 = [], [], [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    w = 0
    with torch.no_grad():
        for x, y, _ in loader:
            P.append(model(x, steps=10).numpy() * S5 + M5)
            T.append(y.numpy() * S5 + M5)
            Pe.append(x[:, -1, 0:6].unsqueeze(1).repeat(1, 10, 1, 1, 1).numpy() * S5 + M5)
            for _ in range(x.shape[0]):
                s = w * 4
                C.append(np.stack([clim_for(int(dts[s + 3 + k])) for k in range(10)]))
                D0.append(int(dts[s + 2]))
                w += 1
    return ds, np.concatenate(P), np.concatenate(T), np.concatenate(Pe), np.array(C, "f4"), np.array(D0)


def skill_block(sc):
    fs, dm = sc["forecast_skill"], sc["dynamics_multiverse"]
    return {
        "windows": None,
        "rmse_ss": fs["overall_rmse_skill_score"], "csi_ss": fs["overall_csi_skill_score"],
        "rmse_pts": fs["rmse_skill_pts"], "csi_pts": fs["csi_pts"], "anti_blur_pts": dm["anti_blur_pts"],
        "gate": sc["admission_gate"]["passed"],
        "day4": [sc["admission_gate"]["day4_pred_rmse"], sc["admission_gate"]["day4_persist_rmse"]],
        "ss_by_day": {d: fs["details"][d]["skill_score"] for d in fs["details"]},
        "csi_by_day": {d: fs["details"][d]["csi_10mm"] for d in fs["details"]},
        "var_ratio_day8": dm["mean_day8_variance_ratios"],
    }


# --- A. reproduce the stored scorecards exactly (same function, same 50 windows) ---
t = time.time()
results["repro"] = {s: evaluate_checkpoint_model(str(CK / "model.pt"), num_eval_windows=50, split=s)["total_score"] for s in ("val", "test")}
results["repro_secs"] = time.time() - t

# --- B. all windows + seasonal breakdown ---
results["full"] = {}
shock_inputs = {}
for split in ("val", "test"):
    ds, P, T, Pe, C, D0 = rollouts(split)
    shock_inputs[split] = (ds, D0)
    months = (D0 // 100) % 100
    blocks = {"all": np.ones(len(D0), bool), "first50": np.arange(len(D0)) < 50,
              "JJAS": np.isin(months, [6, 7, 8, 9]), "non_JJAS": ~np.isin(months, [6, 7, 8, 9])}
    results["full"][split] = {}
    for name, m in blocks.items():
        sc = evaluate_rollouts(P[m], T[m], Pe[m], C[m], shock_results=None)
        b = skill_block(sc)
        b["windows"] = int(m.sum())
        b["date_range"] = [int(D0[m][0]), int(D0[m][-1])]
        results["full"][split][name] = b

json.dump(results, open(OUT, 'w'), indent=1, default=float)
# --- C. causality across start dates (every 3rd window ~ every 12 days) ---
def shock_pair(x0):
    pw = torch.zeros((1, 6, 32, 36))
    mb = box_mask(BOX_BAY_OF_BENGAL)
    sd_ = [float(v) for v in std]
    mn_ = [float(v) for v in mean]
    pw[0, 0, mb] = 3.0 / sd_[0]
    tc = x0[0, -1, 5, mb] * sd_[5] + mn_[5]
    pw[0, 5, mb] = 0.15 * tc / sd_[5]
    ps = torch.zeros((1, 6, 32, 36))
    ma = box_mask(BOX_ARABIAN_SEA)
    u = x0[0, -1, 2, ma] * sd_[2] + mn_[2]
    v = x0[0, -1, 3, ma] * sd_[3] + mn_[3]
    ps[0, 2, ma] = -0.4 * u / sd_[2]
    ps[0, 3, ma] = -0.4 * v / sd_[3]
    return pw, ps, float(u.mean()), float(v.mean())


results["causality"] = {}
for split in ("val", "test"):
    ds, D0 = shock_inputs[split]
    rows = []
    with torch.no_grad():
        for w in range(0, len(ds), 3):
            x0 = ds[w][0]
            x0 = torch.as_tensor(x0).unsqueeze(0)
            base = model(x0, steps=10).numpy()[0] * std[None, :, None, None]
            null = model(x0, steps=10, perturbation=torch.zeros((1, 6, 32, 36))).numpy()[0] * std[None, :, None, None]
            pw, ps, u_as, v_as = shock_pair(x0)
            cw = causality_checks(base, model(x0, steps=10, perturbation=pw).numpy()[0] * std[None, :, None, None], BOX_BAY_OF_BENGAL, std)
            cs = causality_checks(base, model(x0, steps=10, perturbation=ps).numpy()[0] * std[None, :, None, None], BOX_ARABIAN_SEA, std)
            rows.append({"t0": int(D0[w]), "null_max_std": float(np.abs((null - base) / std[None, :, None, None]).max()),
                         "warm": {k: cw[k] for k in ("localized", "propagates", "no_teleport", "stable", "near_frac_day1")},
                         "stall": {k: cs[k] for k in ("localized", "propagates", "no_teleport", "stable", "near_frac_day1")},
                         "as_box_u10": u_as, "as_box_v10": v_as})
    results["causality"][split] = rows

json.dump(results, open(OUT, 'w'), indent=1, default=float)
# --- D. latency through the production inference path (4 threads) ---
sys.path.insert(0, str(REPO))
from inference import predictor  # noqa: E402
from inference.loader import registry  # noqa: E402

registry.load_artifacts()
results["registry_model_name"] = predictor.get_model_name()
data, dts = splits["val"]
init = data[:3].tolist()
lat = []
for k in range(25):
    t = time.perf_counter()
    out, ms = predictor.simulate(init, [0.0, 1.0], 10, None)
    lat.append((time.perf_counter() - t) * 1000)
results["latency_ms"] = {"n": len(lat), "first": lat[0], "median": float(np.median(lat[1:])), "p95": float(np.percentile(lat[1:], 95)),
                         "max": float(np.max(lat[1:])), "out_shape": list(np.asarray(out).shape),
                         "finite": bool(np.isfinite(out).all()), "tp_min": float(np.asarray(out)[:, 1].min()),
                         "t2m_range": [float(np.asarray(out)[:, 0].min()), float(np.asarray(out)[:, 0].max())],
                         "threads": torch.get_num_threads()}

json.dump(results, open(OUT, "w"), indent=1, default=float)
print("done", OUT)
