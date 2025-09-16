#!/usr/bin/env python3
import os, sys, json, math, argparse, importlib
from typing import Dict, Any, List, Tuple
import numpy as np
import h5py
from tqdm.auto import tqdm
import torch
import yaml

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# ---------- utils ----------
def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

@torch.no_grad()
def robust_norm_qmag(vol_2thw: torch.Tensor, q=0.995, target=0.90, max_gain=1.5) -> torch.Tensor:
    vr, vi = vol_2thw[0], vol_2thw[1]
    mag = torch.sqrt(vr*vr + vi*vi)
    qq = torch.quantile(mag.reshape(-1), float(q))
    if not torch.isfinite(qq) or qq <= 1e-6:  # degenerate → return as-is
        return vol_2thw
    g = min(target / max(qq, 1e-6), max_gain)
    m = vol_2thw.abs().amax()
    if m > 0: g = min(g, 1.0 / m)
    return vol_2thw * g

@torch.no_grad()
def patchify2d_fixed_stride(img_hw: torch.Tensor, ph: int, pw: int, sh: int, sw: int) -> torch.Tensor:
    H, W = int(img_hw.shape[-2]), int(img_hw.shape[-1])
    n1 = max(1, math.ceil((H - ph) / sh) + 1)
    n2 = max(1, math.ceil((W - pw) / sw) + 1)
    P = n1 * n2
    out = torch.zeros((P, ph, pw), dtype=img_hw.dtype, device=img_hw.device)
    idx = 0
    for j in range(n1):
        y0 = j * sh; y1 = min(y0 + ph, H); s1 = y1 - y0
        for k in range(n2):
            x0 = k * sw; x1 = min(x0 + pw, W); s2 = x1 - x0
            patch = torch.zeros((ph, pw), dtype=img_hw.dtype, device=img_hw.device)
            if s1 > 0 and s2 > 0:
                patch[:s1, :s2] = img_hw[y0:y1, x0:x1]
            out[idx] = patch; idx += 1
    return out  # [P,ph,pw]

def temporal_starts_full_coverage(T_total: int, T_win: int, prefer_stride: int) -> List[int]:
    if T_total <= T_win: return [0]
    s = max(1, int(prefer_stride))  # normally T-1 → 1-frame overlap
    starts = list(range(0, T_total - T_win + 1, s))
    if starts[-1] + T_win < T_total:
        # recompute mildy-tighter stride so the last window lands on the end
        n = len(starts) + 1
        s = max(1, math.floor((T_total - T_win) / (n - 1)))
        starts = [i * s for i in range(n - 1)]
        last = T_total - T_win
        if starts[-1] != last:
            starts.append(last)
    return starts

def scan_raw_shards(root: str, split: str):
    split_dir = os.path.join(root, split, "shards")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing {split_dir}")
    paths = [os.path.join(split_dir, fn) for fn in sorted(os.listdir(split_dir)) if fn.endswith(".h5")]
    if not paths: raise RuntimeError(f"No .h5 shards under {split_dir}")
    return paths

def write_latent_item(hf: h5py.File, name: str, z: np.ndarray, attrs: Dict[str, Any]):
    g = hf.create_group(name)
    g.create_dataset("z", data=z, dtype="float32", compression="lzf", chunks=True)
    for k, v in attrs.items(): g.attrs[k] = v

# ---------- main ----------
@torch.no_grad()
def process_split(
    in_root: str, out_root: str, split: str,
    vae_import: str, vae_class: str, vae_ckpt: str, strict_load: bool,
    patch_t: int, patch_h: int, patch_w: int,
    stride_h: int, stride_w: int, prefer_t_stride: int,
    device: str, batch_chunks: int, cfg: Dict[str, Any]
):
    assert patch_t % 2 == 1, "patch_t must be odd (e.g., 7)"

    # load frozen VAE
    VAE = dynamic_import(vae_import, vae_class)
    vae = VAE(**cfg["model"]["args"]).to(device).eval()
    if vae_ckpt and os.path.isfile(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        state = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items() }
        vae.load_state_dict(state, strict=strict_load)
        print(f"[VAE] loaded {vae_ckpt}")

    # infer latent H',W' once
    dummy = torch.zeros(2, patch_t, patch_h, patch_w, device=device)
    out_vae = vae([dummy], op="encode")  # list of (mu, logv)
    mu, _ = out_vae[0]
    H_lat, W_lat = int(mu.shape[-2]), int(mu.shape[-1])
    Cz = int(mu.shape[-4])

    in_shards = scan_raw_shards(in_root, split)

    # outputs
    out_dir = os.path.join(out_root, split); ensure_dir(out_dir)
    out_shards = os.path.join(out_dir, "latent_shards"); ensure_dir(out_shards)
    meta_path = os.path.join(out_dir, "latents_meta.json")
    meta = {
        "split": split,
        "note": "deterministic VAE latents per 7-frame window",
        "latent_shape": {"Cz": Cz, "T": patch_t, "H": H_lat, "W": W_lat},
        "patch": {"t": patch_t, "h": patch_h, "w": patch_w},
        "spatial_stride": {"h": (stride_h if stride_h>0 else patch_h//2),
                           "w": (stride_w if stride_w>0 else patch_w//2)},
        "temporal_stride_pref": prefer_t_stride,
        "items": []
    }

    shard_size = 2000
    item_counter, shard_idx = 0, 0
    hf, cur_path = None, None

    def roll():
        nonlocal hf, shard_idx, cur_path
        if hf is not None: hf.close()
        cur_path = os.path.join(out_shards, f"latent_{shard_idx:05d}.h5")
        hf = h5py.File(cur_path, "w", libver="latest")
        shard_idx += 1

    roll()
    print("TOTAL NUMBER OF SHARDS: ", len(in_shards))

    for sp in in_shards:
        with h5py.File(sp, "r", libver="latest") as fr:
            for key in tqdm(list(fr["volumes"].keys()), desc=f"{split}:{os.path.basename(sp)}", leave=False):
                x = torch.from_numpy(fr["volumes"][key][()]).float()  # [2,T,H,W] normalized by your earlier preproc
                x = robust_norm_qmag(x)  # keep behavior consistent with “normalize: qmag”
                if x.shape[1] % 2 == 0:
                    x = x[:, :x.shape[1]-1]  # force odd T to match prior usage

                _, Ttot, H, W = x.shape
                sh = max(1, stride_h if stride_h>0 else patch_h//2)
                sw = max(1, stride_w if stride_w>0 else patch_w//2)
                t_starts = temporal_starts_full_coverage(Ttot, patch_t, prefer_t_stride)

                # how many spatial patches?
                n1 = max(1, math.ceil((H - patch_h) / sh) + 1)
                n2 = max(1, math.ceil((W - patch_w) / sw) + 1)
                P_expected = n1 * n2

                for t0 in t_starts:
                    win = x[:, t0:t0+patch_t].to(device)  # [2,7,H,W]

                    # spatial patches per frame (row-major order exactly like patchify2d_fixed_stride)
                    r_list = [patchify2d_fixed_stride(win[0, t], patch_h, patch_w, sh, sw) for t in range(patch_t)]
                    i_list = [patchify2d_fixed_stride(win[1, t], patch_h, patch_w, sh, sw) for t in range(patch_t)]

                    P = P_expected  # enforce
                    clips = []
                    for p in range(P):
                        clip = torch.zeros(2, patch_t, patch_h, patch_w, device=device)
                        for t in range(patch_t):
                            clip[0, t] = r_list[t][p]
                            clip[1, t] = i_list[t][p]
                        clips.append(clip.unsqueeze(0))  # [1,2,7,ph,pw]

                    # encode in micro-batches, take deterministic z = mu
                    zs = []
                    for i in range(0, P, batch_chunks):
                        chunk = clips[i:i+batch_chunks]
                        mus, _ = zip(*vae(chunk, op="encode"))  # each [1,Cz,7,H',W']
                        for m in mus: zs.append(m.squeeze(0).cpu())
                    # [P,Cz,7,H',W'] → [Cz,7,P,H',W']
                    z = torch.stack(zs, dim=0).permute(1,2,0,3,4).contiguous()

                    name = f"{item_counter:08d}"
                    write_latent_item(hf, name, z.numpy(), {
                        "source_shard": os.path.basename(sp), "source_key": key,
                        "t0": int(t0), "P": int(P_expected),
                        "ph": int(patch_h), "pw": int(patch_w),
                        "sh": int(sh), "sw": int(sw),
                        "n1": int(n1), "n2": int(n2),
                        "H": int(H), "W": int(W),
                    })

                    meta["items"].append({
                        "ds": os.path.basename(cur_path), "name": name,
                        "source_shard": os.path.basename(sp), "source_key": key,
                        "t0": int(t0), "P": int(P_expected)
                    })
                    item_counter += 1
                    if item_counter % shard_size == 0: roll()

    if hf is not None: hf.close()
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    print(f"[done] wrote {item_counter} latent windows  →  {meta_path}")

def main():
    ap = argparse.ArgumentParser("Precompute 7-frame latent windows (VAE once, FM trains VAE-free).")
    ap.add_argument("--config", default="configs/vae.yaml")
    ap.add_argument("--in_root",  default="/storage/CINE_data", help="Raw CINE root (…/train|val|test/shards/*.h5)")
    ap.add_argument("--out_root", default="/storage/CINE_data/latents", help="Output root for latent windows")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--vae_import", default="CardiacVAE.model.vae")
    ap.add_argument("--vae_class",  default="CardiacVAE")
    ap.add_argument("--vae_ckpt",   default="/storage/matt_models/cardiac_vae/videos/step_0195000/state.pt")
    ap.add_argument("--strict_load", action="store_true")
    ap.add_argument("--patch_t", type=int, default=7)
    ap.add_argument("--patch_h", type=int, default=80)
    ap.add_argument("--patch_w", type=int, default=80)
    ap.add_argument("--stride_h", type=int, default=0, help="0→ph//2")
    ap.add_argument("--stride_w", type=int, default=0, help="0→pw//2")
    ap.add_argument("--prefer_t_stride", type=int, default=6, help="T-1 → 1-frame overlap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_chunks", type=int, default=64)
    ap.add_argument("--overlap", type=float, default=0.1,
                    help="Spatial overlap fraction in [0,1). If set and stride_h/w==0: stride = patch*(1-overlap).")
    args = ap.parse_args()

    if args.stride_h == 0 and (args.overlap is not None):
        args.stride_h = max(1, int(round(args.patch_h * (1.0 - float(args.overlap)))))
    if args.stride_w == 0 and (args.overlap is not None):
        args.stride_w = max(1, int(round(args.patch_w * (1.0 - float(args.overlap)))))

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    process_split(
        args.in_root, args.out_root, args.split,
        args.vae_import, args.vae_class, args.vae_ckpt, args.strict_load,
        args.patch_t, args.patch_h, args.patch_w,
        args.stride_h, args.stride_w, args.prefer_t_stride,
        args.device, args.batch_chunks, cfg
    )

if __name__ == "__main__":
    main()
