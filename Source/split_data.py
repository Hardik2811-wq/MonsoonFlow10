"""
Chronological train / val / test split for Track 2 (Multiverse of Monsoon).

Default split (calendar years, never random):
    train : 2005-2018   -> fit model, normalisation stats, climatology
    val   : 2019-2020   -> model selection, early stopping, tuning
    test  : 2021        -> sealed; run ONCE after the model is frozen

Usage:
    python split_data.py --input path/to/india_climate_train.nc --out data/splits

Outputs (in --out):
    train.npz, val.npz, test.npz   states [T, 6, 32, 36] float32, dates, doy
    static.npy                     [2, 32, 36] land_sea_mask, orography_m
    train_stats.json               per-channel mean/std from TRAIN ONLY
    train_climatology.npy          [366, 6, 32, 36] from TRAIN ONLY (15-day smoothed)
    splits.json                    manifest: years, day counts, window counts

Windows (3 input days + up to 10 target days) are built inside one split only,
so no target day ever crosses into another split. Use --gap-days to also drop
days at the start of val/test for extra separation from autocorrelation.

Do NOT commit the .nc file or these .npz files to the submission repo
(repo_scan flags raw climate data and multi-year daily archives).
Ship only weights, stats and static fields.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
STATIC = ["land_sea_mask", "orography_m"]
T_IN = 3          # input days per request (contract)
ROLLOUT = 10      # scored rollout steps (contract)
NLAT, NLON = 32, 36


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def find_time_dim(ds):
    for name in ("time", "date", "valid_time", "t"):
        if name in ds.dims:
            return name
    for d in ds.dims:
        if np.issubdtype(ds[d].dtype, np.datetime64):
            return d
    raise ValueError(f"Could not find a time dimension in {list(ds.dims)}")


def load_dynamic(ds, tdim):
    """Return [T, 6, 32, 36] float32 in CHANNELS order, whatever the file layout."""
    # Layout A: one variable per channel (t2m, tp, ...)
    if all(c in ds.data_vars for c in CHANNELS):
        arrs = [ds[c].transpose(tdim, ...).values for c in CHANNELS]
        return np.stack(arrs, axis=1).astype(np.float32)

    # Layout B: one variable with a channel dimension
    for name, da in ds.data_vars.items():
        if tdim in da.dims and da.ndim == 4:
            cdim = [d for d in da.dims if d != tdim and da.sizes[d] == len(CHANNELS)]
            if not cdim:
                continue
            cdim = cdim[0]
            if cdim in da.coords and da[cdim].dtype.kind in "US":
                names = [str(v) for v in da[cdim].values]
                if names != CHANNELS:
                    print(f"  reordering channels {names} -> {CHANNELS}")
                    da = da.sel({cdim: CHANNELS})
            spatial = [d for d in da.dims if d not in (tdim, cdim)]
            out = da.transpose(tdim, cdim, *spatial).values.astype(np.float32)
            print(f"  using variable '{name}' with dims {da.dims}")
            return out
    raise ValueError(
        "Could not locate the 6 dynamic channels. Variables found: "
        f"{ {k: v.dims for k, v in ds.data_vars.items()} }"
    )


def load_static(ds):
    if "static" in ds.data_vars:
        return ds["static"].values.astype(np.float32)
    if all(s in ds.data_vars for s in STATIC):
        return np.stack([ds[s].values for s in STATIC]).astype(np.float32)
    print("  WARNING: static fields not found; static.npy not written")
    return None


def count_windows(n_days, horizon):
    return max(0, n_days - T_IN - horizon + 1)


def climatology(states, doy, smooth_days=15):
    """[366, C, H, W] mean by day-of-year, circularly smoothed (like the kit)."""
    C, H, W = states.shape[1:]
    sums = np.zeros((366, C, H, W), np.float64)
    counts = np.zeros(366, np.int64)
    for i, d in enumerate(doy):
        sums[d - 1] += states[i]
        counts[d - 1] += 1
    # day 366 only exists in leap years; borrow day 365 if it is empty
    if counts[365] == 0:
        sums[365], counts[365] = sums[364], counts[364]
    raw = sums / np.maximum(counts, 1)[:, None, None, None]
    half = smooth_days // 2
    out = np.zeros_like(raw)
    for k in range(-half, half + 1):
        out += np.roll(raw, k, axis=0)
    return (out / (2 * half + 1)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="path to india_climate_train.nc")
    p.add_argument("--out", default="data/splits")
    p.add_argument("--train-years", default="2005-2018")
    p.add_argument("--val-years", default="2019-2020")
    p.add_argument("--test-years", default="2021")
    p.add_argument("--gap-days", type=int, default=0,
                   help="drop this many days at the start of val and test (extra separation)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.input}")
    ds = xr.open_dataset(args.input)
    tdim = find_time_dim(ds)
    times = ds[tdim].values.astype("datetime64[D]")
    states = load_dynamic(ds, tdim)
    static = load_static(ds)

    # ---- sanity checks -------------------------------------------------------
    assert states.shape[1:] == (6, NLAT, NLON), f"unexpected shape {states.shape}"
    order = np.argsort(times)
    if not np.all(order == np.arange(len(times))):
        print("  WARNING: time axis was not sorted; sorting")
        times, states = times[order], states[order]
    gaps = np.diff(times).astype(int)
    if np.any(gaps != 1):
        bad = np.where(gaps != 1)[0][:5]
        print(f"  WARNING: time axis has {np.sum(gaps != 1)} non-daily steps, e.g. after {times[bad]}")
    n_nan = int(np.isnan(states).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} NaN values in dynamic channels")
    print(f"  {len(times)} days, {times[0]} .. {times[-1]}, shape {states.shape}")

    years = times.astype("datetime64[Y]").astype(int) + 1970
    doy = (times - times.astype("datetime64[Y]")).astype(int) + 1

    split_years = {
        "train": parse_years(args.train_years),
        "val": parse_years(args.val_years),
        "test": parse_years(args.test_years),
    }
    all_y = sum(split_years.values(), [])
    assert len(all_y) == len(set(all_y)), "a year appears in more than one split"
    assert max(split_years["train"]) < min(split_years["val"]) <= max(split_years["val"]) < min(split_years["test"]), \
        "splits must be chronological: train < val < test"
    missing = sorted(set(years) - set(all_y))
    if missing:
        print(f"  NOTE: years in file but not in any split: {missing}")

    manifest = {"source": str(args.input), "channels": CHANNELS, "gap_days": args.gap_days, "splits": {}}
    split_data = {}
    for name, ys in split_years.items():
        idx = np.where(np.isin(years, ys))[0]
        if name != "train" and args.gap_days:
            idx = idx[args.gap_days:]
        if len(idx) == 0:
            raise ValueError(f"split '{name}' is empty for years {ys}")
        s = states[idx]
        split_data[name] = (s, doy[idx])
        np.savez(out / f"{name}.npz", states=s, dates=times[idx].astype(str), doy=doy[idx])
        manifest["splits"][name] = {
            "years": ys,
            "first_day": str(times[idx[0]]),
            "last_day": str(times[idx[-1]]),
            "n_days": int(len(idx)),
            "n_windows_1step": count_windows(len(idx), 1),
            "n_windows_10step": count_windows(len(idx), ROLLOUT),
        }
        print(f"  {name:5s}: {times[idx[0]]} .. {times[idx[-1]]}  "
              f"{len(idx)} days, {count_windows(len(idx), ROLLOUT)} ten-day windows")

    # ---- train-only statistics ----------------------------------------------
    tr, tr_doy = split_data["train"]
    stats = {
        c: {"mean": float(np.nanmean(tr[:, i])), "std": float(np.nanstd(tr[:, i]))}
        for i, c in enumerate(CHANNELS)
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    np.save(out / "train_climatology.npy", climatology(tr, tr_doy))
    if static is not None:
        np.save(out / "static.npy", static)

    (out / "splits.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote splits, train_stats.json and train_climatology.npy to {out}/")


if __name__ == "__main__":
    main()
