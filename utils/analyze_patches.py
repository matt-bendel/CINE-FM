#!/usr/bin/env python3
# tools/analyze_patches.py
import os, sys, json, math, argparse, yaml, pathlib, random
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- import your dataset exactly like training ---
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)
from data.cine_dataset import CINEDataset   # uses the same patch logic as training

# ---------- dataloader helpers (same defaults as training) ----------
def ragged_collate(batch):
    return batch

def build_train_loader(ds_cfg: Dict[str, Any], dl_cfg: Dict[str, Any]) -> DataLoader:
    name = ds_cfg["name"]; args = ds_cfg.get("args", {})
    if name not in ("CINEDataset",):
        raise ValueError("Adjust build_train_loader: expected CINEDataset.")
    dataset = CINEDataset(**args)  # returns [2,1,64,64] or [2,11,64,64] for TRAIN
    bsz = int(dl_cfg["train_batch_size"])
    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=dl_cfg.get("shuffle", True),
        num_workers=int(dl_cfg.get("num_workers", 4)),
        pin_memory=bool(dl_cfg.get("pin_memory", True)),
        drop_last=True,
        collate_fn=ragged_collate,
    )

# ---------- reservoir-lite sampler (fills once; dataloader is shuffled) ----------
class SamplerBuf:
    def __init__(self, cap: int):
        self.cap = int(cap)
        self.buf = None
        self.n = 0

    def add(self, x_flat_cpu_float: torch.Tensor):
        if self.cap <= 0: return
        x = x_flat_cpu_float
        if self.buf is None:
            self.buf = torch.empty((0,), dtype=torch.float32)
        remaining = self.cap - self.buf.numel()
        if remaining <= 0:
            return
        take = min(remaining, x.numel())
        if take > 0:
            # pick a random subset from x
            idx = torch.randperm(x.numel())[:take]
            self.buf = torch.cat([self.buf, x[idx].contiguous().view(-1)], dim=0)
        self.n += x.numel()

    def numpy(self):
        return self.buf.detach().cpu().numpy() if (self.buf is not None and self.buf.numel() > 0) else np.array([])

# ---------- plotting ----------
def save_hist(data: np.ndarray, title: str, path: str, bins=256, range=None, logy=False):
    plt.figure(figsize=(6,4), dpi=120)
    plt.hist(data, bins=bins, range=range, alpha=0.85, edgecolor='none', log=logy)
    plt.title(title)
    plt.xlabel('value'); plt.ylabel('count (log)' if logy else 'count')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_hist_abs(data: np.ndarray, title: str, path: str, bins=256, logy=False):
    save_hist(np.abs(data), title + " (abs)", path, bins=bins, range=None, logy=logy)

# ---------- main ----------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=200, help="how many training batches to scan")
    parser.add_argument("--sample-cap", type=int, default=1_000_000, help="samples kept for hist/quantiles per signal")
    parser.add_argument("--out", type=str, default="out/analyze_patches")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- load config & dataloader (TRAIN) exactly like your training script ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_dl = build_train_loader(cfg["train_dataset"], cfg["dataloader"])
    stage_mode = cfg["stages"]["mode"]  # "pretrain_2d" or "videos"

    # --- running stats (global, exact) ---
    device = torch.device("cpu")
    stats = {
        "r":  {"sum":0.0, "sumsq":0.0, "min": float("+inf"), "max": float("-inf"), "count": 0, "sat_pos":0, "sat_neg":0, "gt1":0},
        "i":  {"sum":0.0, "sumsq":0.0, "min": float("+inf"), "max": float("-inf"), "count": 0, "sat_pos":0, "sat_neg":0, "gt1":0},
        "mag":{"sum":0.0, "sumsq":0.0, "min": float("+inf"), "max": float("-inf"), "count": 0, "gt_sqrt2":0},
    }
    # per-patch magnitude mean/std distributions
    patch_mag_means = SamplerBuf(cap=min(args.sample_cap//64, 250_000))
    patch_mag_stds  = SamplerBuf(cap=min(args.sample_cap//64, 250_000))

    # sample buffers (approximate quantiles / hist)
    buf_r   = SamplerBuf(cap=args.sample_cap//2)
    buf_i   = SamplerBuf(cap=args.sample_cap//2)
    buf_mag = SamplerBuf(cap=args.sample_cap)

    batches_scanned = 0
    pbar = tqdm(total=args.max_batches, desc="scan-train", dynamic_ncols=True)

    for batch_list in train_dl:
        batches_scanned += 1
        for x in batch_list:
            # x: [2, T, H, W] (patches in training)
            x = x.to(torch.float32, copy=False)
            r, im = x[0], x[1]  # [T,H,W]
            mag = torch.sqrt(torch.clamp(r*r + im*im, min=0.0) + 1e-12)

            # --- global exact min/max/sum/sumsq & saturation counters ---
            for name, t, thr in (("r", r, 1.0), ("i", im, 1.0)):
                stats[name]["min"] = min(stats[name]["min"], float(t.min()))
                stats[name]["max"] = max(stats[name]["max"], float(t.max()))
                stats[name]["sum"]   += float(t.sum())
                stats[name]["sumsq"] += float((t*t).sum())
                stats[name]["count"] += t.numel()
                stats[name]["sat_pos"] += int((t >  (thr-1e-6)).sum())
                stats[name]["sat_neg"] += int((t < -(thr-1e-6)).sum())
                stats[name]["gt1"]    += int((t.abs() > thr).sum())

            stats["mag"]["min"] = min(stats["mag"]["min"], float(mag.min()))
            stats["mag"]["max"] = max(stats["mag"]["max"], float(mag.max()))
            stats["mag"]["sum"]   += float(mag.sum())
            stats["mag"]["sumsq"] += float((mag*mag).sum())
            stats["mag"]["count"] += mag.numel()
            stats["mag"]["gt_sqrt2"] += int((mag > math.sqrt(2.0)).sum())

            # --- sampling for quantiles/histograms (random subset) ---
            # dataloader is shuffled, so taking the first K samples is fine in practice
            buf_r.add(r.flatten().cpu())
            buf_i.add(im.flatten().cpu())
            buf_mag.add(mag.flatten().cpu())

            # per-patch magnitude mean/std (aggregated over all voxels in the patch)
            patch_mag_means.add(mag.mean().view(-1).cpu())
            patch_mag_stds.add(mag.std(unbiased=False).view(-1).cpu())

        pbar.update(1)
        if batches_scanned >= args.max_batches:
            break
    pbar.close()

    # --- summarize ---
    def finalize_axis(d: Dict[str, Any], name: str):
        n = max(1, d["count"])
        mean = d["sum"]/n
        var  = max(0.0, d["sumsq"]/n - mean*mean)
        std  = math.sqrt(var)
        out  = {
            "count": n,
            "min": d["min"],
            "max": d["max"],
            "mean": mean,
            "std": std,
        }
        if name in ("r","i"):
            out.update({
                "frac_gt_1": d["gt1"]/n,
                "frac_sat_pos_(>=0.999)": d["sat_pos"]/n,
                "frac_sat_neg_(<=-0.999)": d["sat_neg"]/n,
            })
        if name == "mag":
            out.update({"frac_gt_sqrt2": d["gt_sqrt2"]/n})
        return out

    summary = {
        "stage_mode": stage_mode,
        "batches_scanned": batches_scanned,
        "r":   finalize_axis(stats["r"], "r"),
        "i":   finalize_axis(stats["i"], "i"),
        "mag": finalize_axis(stats["mag"], "mag"),
    }

    # --- quantiles from samples ---
    def qdict(arr: np.ndarray, qs=(0.001,0.01,0.05,0.5,0.95,0.99,0.995,0.999)):
        if arr.size == 0:
            return {}
        vals = np.quantile(arr, qs, method="linear")
        return {f"q{int(1000*q)/10:.1f}%": float(v) for q, v in zip(qs, vals)}

    r_np   = buf_r.numpy()
    i_np   = buf_i.numpy()
    mag_np = buf_mag.numpy()
    patch_mean_np = patch_mag_means.numpy()
    patch_std_np  = patch_mag_stds.numpy()

    summary["r_quantiles"]   = qdict(r_np)
    summary["i_quantiles"]   = qdict(i_np)
    summary["mag_quantiles"] = qdict(mag_np)
    summary["patch_mag_mean_quantiles"] = qdict(patch_mean_np)
    summary["patch_mag_std_quantiles"]  = qdict(patch_std_np)

    # --- recommended scales (for LPIPS, fixed scaling, etc.) ---
    # Use a high percentile of magnitude as the "max" for stable mapping.
    if mag_np.size > 0:
        q995  = float(np.quantile(mag_np, 0.995))
        q999  = float(np.quantile(mag_np, 0.999))
        # A conservative suggestion for LPIPS fixed scale:
        # pick between q99.5 and q99.9 with a small headroom.
        lpips_max_mag = 1.05 * q999
        summary["recommendations"] = {
            "lpips_max_mag_suggestion": lpips_max_mag,
            "also_consider": {"q99.5": q995, "q99.9": q999},
            "sqrt2_reference": math.sqrt(2.0),
        }

    # --- write JSON summary ---
    out_json = os.path.join(args.out, "train_patch_stats.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n==== Summary ====")
    print(json.dumps(summary, indent=2))

    # --- plots ---
    if r_np.size:
        save_hist(r_np,   "Real values (train patches)",   os.path.join(args.out, "hist_real.png"), bins=256, logy=True)
        save_hist_abs(r_np, "Real |value|",                os.path.join(args.out, "hist_real_abs.png"), bins=256, logy=True)
    if i_np.size:
        save_hist(i_np,   "Imag values (train patches)",   os.path.join(args.out, "hist_imag.png"), bins=256, logy=True)
        save_hist_abs(i_np, "Imag |value|",                os.path.join(args.out, "hist_imag_abs.png"), bins=256, logy=True)
    if mag_np.size:
        save_hist(mag_np, "Magnitude (train patches)",     os.path.join(args.out, "hist_mag.png"),  bins=256, logy=True)
        save_hist_abs(mag_np, "Magnitude |value|",         os.path.join(args.out, "hist_mag_abs.png"), bins=256, logy=True)
    if patch_mean_np.size:
        save_hist(patch_mean_np, "Per-patch magnitude mean", os.path.join(args.out, "hist_patch_mag_mean.png"), bins=256, logy=True)
    if patch_std_np.size:
        save_hist(patch_std_np,  "Per-patch magnitude std",  os.path.join(args.out, "hist_patch_mag_std.png"), bins=256, logy=True)

    print(f"\nSaved:\n- {out_json}\n- PNG histograms under: {args.out}")

if __name__ == "__main__":
    main()
