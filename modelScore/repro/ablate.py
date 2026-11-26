"""Ablation: MF10 with its neural increment head zeroed (= damped-anomaly + transport prior). Read-only."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]  # repo root (modelScore/repro/ -> ../../)
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "resources" / "yug-monsoon-kit"))
from data.preprocessing import load_raw_dataset, chronological_split
from data.dataset import MonsoonDataset
from models.mf10 import MonsoonFlow10
from evaluation.eval_scoring import evaluate_rollouts
torch.set_num_threads(4)
CK = REPO / "checkpoints"; stats = json.load(open(CK / "stats.json")); static = np.load(CK / "static.npy"); clim = np.load(CK / "climatology.npy")
M5 = np.array(stats["mean"], "f4").reshape(1, 1, 6, 1, 1); S5 = np.array(stats["std"], "f4").reshape(1, 1, 6, 1, 1)
def mk(zero_head):
    m = MonsoonFlow10(in_channels=10, dynamic_channels=6, hidden_dim=64, num_lstm_layers=2, stats=stats)
    sd = torch.load(CK / "model.pt", map_location="cpu", weights_only=False); m.load_state_dict(sd.get("state_dict", sd), strict=True)
    if zero_head:
        with torch.no_grad(): m.decoder.out_conv[-1].weight.zero_(); m.decoder.out_conv[-1].bias.zero_()
    return m.eval()
raw, dates = load_raw_dataset(REPO / "data" / "india_climate_train.nc"); splits = chronological_split(raw, dates)
# Row in the 366-row leap-indexed climatology table (no date literals): cumulative leap-year month offsets
LEAP_CUM = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
def cl(d):
    return clim[LEAP_CUM[d // 100 % 100 - 1] + d % 100 - 1]
out = {}
for split in ("val", "test"):
    data, dts = splits[split]; ds = MonsoonDataset(data, dts, stats, static, t_in=3, t_rollout=10, step_stride=4)
    X, Y = zip(*[(ds[i][0], ds[i][1]) for i in range(len(ds))]); X = torch.as_tensor(np.stack(X)); Y = np.stack(Y) * S5 + M5
    Pe = X[:, -1, 0:6].unsqueeze(1).repeat(1, 10, 1, 1, 1).numpy() * S5 + M5
    C = np.array([[cl(int(dts[w * 4 + 3 + k])) for k in range(10)] for w in range(len(ds))], "f4")
    for name, zh in (("full_model", False), ("prior_only_head_zeroed", True)):
        with torch.no_grad(): P = mk(zh)(X, steps=10).numpy() * S5 + M5
        sc = evaluate_rollouts(P, Y, Pe, C, shock_results=None); fs = sc["forecast_skill"]
        out[f"{split}/{name}"] = {"rmse_ss": fs["overall_rmse_skill_score"], "csi_ss": fs["overall_csi_skill_score"], "skill_pts": fs["total_pts"],
                                 "day4": sc["admission_gate"]["day4_pred_rmse"], "blur": sc["dynamics_multiverse"]["anti_blur_pts"]}
        print(split, name, {k: round(v, 4) for k, v in out[f"{split}/{name}"].items()}, flush=True)
json.dump(out, open(Path(__file__).with_name("ablation.json"), "w"), indent=1)
