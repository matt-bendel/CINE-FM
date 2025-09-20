#!/usr/bin/env python3
"""
Build and visualize a GLOBAL latent from a validation video, handling temporal compression (pt != nt).

- Patchify [2,T,H,W] in pixel space (5% spatial overlap; 1-frame temporal overlap in pixels)
- VAE-encode each [2,pt,ph,pw] patch -> [Cz, nt, Hlp, Wlp] (nt may differ from pt)
- Overlap-add in LATENT space spatially per chunk -> [Cz, nt, H', W']
- Place each chunk along the LATENT timeline using s_t = nt/pt:
      t_lat_start = round(t0 * s_t)
      t_lat_end   = t_lat_start + nt
- Average overlaps across time and space to obtain GLOBAL latent [Cz, T_lat, H', W']
- Save a 4×4 grid video of the first 16 channels over latent time

Run:
  python scripts/make_global_latent.py --config configs/flow_matching.yaml \
      --video-index 0 --outdir ./_global_latent
"""

import os, sys, math, argparse, importlib, yaml
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ---------------- utils ----------------
def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def try_import_dataset(name: str):
    if name == "CINEFlowMatchLatentDataset":
        mod = importlib.import_module("data.cine_flow_dataset")
        return getattr(mod, "CINEFlowMatchLatentDataset")
    elif name in ("CINEDataset", "CINEFlowMatchDataset"):
        mod = importlib.import_module("data.cine_dataset")
        return getattr(mod, "CINEDataset")
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

def frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if f.numel() == 0:
        return np.zeros((1,1), dtype=np.uint8)
    flat = f.flatten()
    lo = torch.quantile(flat, lo_p)
    hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

def save_video_gray(frames_uint8: List[np.ndarray], path: str, fps: int = 7):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio is None:
        base = os.path.splitext(path)[0]
        for i, fr in enumerate(frames_uint8):
            from imageio import imwrite
            imwrite(f"{base}_frame_{i:04d}.png", fr)
        print(f"[warn] wrote PNG frames to {base}_frame_*.png (no ffmpeg).")
        return
    try:
        rgb_frames = [np.repeat(f[None, ...], 3, axis=0).transpose(1,2,0) for f in frames_uint8]
        imageio.mimsave(path, rgb_frames, fps=fps, codec="libx264", quality=8)
        print(f"[ok] wrote {path}")
    except Exception as e:
        gif_path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, frames_uint8, fps=fps)
        print(f"[ok] wrote {gif_path} (mp4 failed: {e})")

# -------- patchify/depatchify over time (spatial only) --------
def pct_to_stride_len(P: int, pct: float) -> int:
    ov = max(0.0, min(99.0, float(pct))) / 100.0
    return max(1, int(math.ceil(P * (1.0 - ov))))

def spatial_coords(H: int, W: int, ph: int, pw: int, sh: int, sw: int):
    n1 = max(1, math.ceil((H - ph) / sh) + 1)
    n2 = max(1, math.ceil((W - pw) / sw) + 1)
    coords = []
    for j in range(n1):
        y0 = j * sh; y1 = min(y0 + ph, H); s1 = y1 - y0
        for k in range(n2):
            x0 = k * sw; x1 = min(x0 + pw, W); s2 = x1 - x0
            coords.append((y0, y1, s1, x0, x1, s2, j, k, n1, n2))
    return coords

def spatial_patchify_video(x_2thw: torch.Tensor, ph: int, pw: int, sh: int, sw: int, coords):
    device = x_2thw.device; dtype = x_2thw.dtype
    _, T, H, W = x_2thw.shape
    P = len(coords)
    out = torch.zeros((P, 2, T, ph, pw), dtype=dtype, device=device)
    for idx, (y0,y1,s1,x0,x1,s2, *_rest) in enumerate(coords):
        patch = torch.zeros((2, T, ph, pw), dtype=dtype, device=device)
        if s1 > 0 and s2 > 0:
            patch[:, :, :s1, :s2] = x_2thw[:, :, y0:y1, x0:x1]
        out[idx] = patch
    return out

def _axis_weights(L_eff: int, idx: int, n: int, O: int, device):
    has_prev = (idx > 0); has_next = (idx < n - 1)
    L_left  = min(O if has_prev else 0, L_eff)
    L_right = min(O if has_next else 0, L_eff)
    if L_left + L_right > L_eff:
        if L_left > 0 and L_right > 0:
            tot = L_left + L_right
            L_left_new  = max(1, int(round(L_eff * (L_left / tot))))
            L_right_new = L_eff - L_left_new
            L_left, L_right = L_left_new, L_right_new
        else:
            L_left  = min(L_left,  L_eff)
            L_right = L_eff - L_left
    w = torch.ones(L_eff, dtype=torch.float32, device=device)
    if L_left > 0:
        w[:L_left] = 0.5 if L_left == 1 else torch.linspace(0.0, 1.0, steps=L_left, device=device)
    if L_right > 0:
        w[-L_right:] = 0.5 if L_right == 1 else torch.linspace(1.0, 0.0, steps=L_right, device=device)
    return w

def temporal_chunk_starts(T: int, pt: int) -> List[int]:
    if pt >= T: return [0]
    step = max(1, pt - 1)  # 1-frame overlap (pixel domain)
    starts = list(range(0, T - pt + 1, step))
    if starts[-1] != T - pt:
        starts.append(T - pt)
    return starts

# -------------- VAE wrappers --------------
@torch.no_grad()
def encode_pixel_patches(vae, patches_P2thw: torch.Tensor, bs_vae: int) -> torch.Tensor:
    outs = []
    P = patches_P2thw.shape[0]
    for i in range(0, P, bs_vae):
        chunk = patches_P2thw[i:i+bs_vae]
        x_list = [x for x in chunk]                      # list of [2,pt,ph,pw]
        pairs = vae(x_list, op="encode")                 # list of (mu, logv) or [mu]
        for pr in pairs:
            mu = pr[0] if isinstance(pr, (list, tuple)) else pr
            if mu.dim() == 5 and mu.shape[0] == 1:
                mu = mu.squeeze(0)                       # [Cz, nt, Hlp, Wlp]
            outs.append(mu)
    return torch.stack(outs, dim=0)                       # [P, Cz, nt, Hlp, Wlp]

# -------------- latent depatchify (spatial only, per chunk) --------------
@torch.no_grad()
def depatchify_latent_from_pixel_grid(
    z_patches_Pcnhw: torch.Tensor,
    H_px: int, W_px: int,
    ph_px: int, pw_px: int,
    sh_px: int, sw_px: int,
    coords_px,
) -> torch.Tensor:
    """
    Overlap-add in LATENT space (spatial dims only) for *one temporal chunk*.
    Returns [Cz, nt, H_lat, W_lat].
    """
    device = z_patches_Pcnhw.device
    P, Cz, nt, Hlp, Wlp = z_patches_Pcnhw.shape

    s_h = Hlp / float(ph_px)
    s_w = Wlp / float(pw_px)
    H_lat = int(math.ceil(H_px * s_h))
    W_lat = int(math.ceil(W_px * s_w))

    sh_lat = max(1, int(round(sh_px * s_h)))
    sw_lat = max(1, int(round(sw_px * s_w)))
    O1_lat = max(0, Hlp - sh_lat)
    O2_lat = max(0, Wlp - sw_lat)

    out_num = torch.zeros((Cz, nt, H_lat, W_lat), dtype=z_patches_Pcnhw.dtype, device=device)
    out_den = torch.zeros((1, 1, H_lat, W_lat), dtype=torch.float32, device=device)

    n1 = coords_px[0][8]; n2 = coords_px[0][9]
    idx = 0
    for (y0,y1,s1,x0,x1,s2,j,k, *_rest) in coords_px:
        s1_lat = min(Hlp, H_lat - j * sh_lat)
        s2_lat = min(Wlp, W_lat - k * sw_lat)
        if s1_lat <= 0 or s2_lat <= 0:
            idx += 1
            continue
        y0_lat = j * sh_lat; x0_lat = k * sw_lat
        y1_lat = y0_lat + s1_lat; x1_lat = x0_lat + s2_lat

        w1 = _axis_weights(s1_lat, j, n1, O1_lat, device)
        w2 = _axis_weights(s2_lat, k, n2, O2_lat, device)
        w = (w1[None, None, :, None] * w2[None, None, None, :])  # [1,1,s1_lat,s2_lat]

        p = z_patches_Pcnhw[idx][:, :, :s1_lat, :s2_lat]          # [Cz, nt, s1_lat, s2_lat]
        out_num[:, :, y0_lat:y1_lat, x0_lat:x1_lat] += (p * w).to(out_num.dtype)
        out_den[:, :, y0_lat:y1_lat, x0_lat:x1_lat] += w
        idx += 1

    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)                    # [Cz, nt, H_lat, W_lat]

# -------------- viz: 4×4 channel grid video over LATENT time --------------
@torch.no_grad()
def save_latent_grid_video(Z_cThw: torch.Tensor, path: str, fps: int = 7, max_channels: int = 16):
    """
    Z_cThw: [Cz, T_lat, H', W'] (float)
    """
    Cz, T, H, W = Z_cThw.shape
    Cuse = min(max_channels, Cz)
    rows = cols = 4 if Cuse >= 16 else int(math.ceil(math.sqrt(Cuse)))

    # robust per-channel scaling across the whole latent timeline
    Z = Z_cThw.detach().float()
    Zf = Z.reshape(Cz, -1)
    lo = torch.quantile(Zf, 0.01, dim=1)
    hi = torch.quantile(Zf, 0.99, dim=1)
    lo = lo[:Cuse].view(Cuse, 1, 1)
    hi = hi[:Cuse].view(Cuse, 1, 1)

    frames = []
    for t in range(T):
        tiles = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                if idx < Cuse:
                    ch = Z[idx, t]                                 # [H,W]
                    g = (ch - lo[idx]) / (hi[idx] - lo[idx] + 1e-8)
                    g = g.clamp_(0, 1).unsqueeze(0)
                    row_tiles.append(g)
                else:
                    row_tiles.append(torch.zeros(1, H, W))
            tiles.append(torch.cat(row_tiles, dim=-1))
        grid = torch.cat(tiles, dim=-2)                            # [1, rows*H, cols*W]
        frames.append(frame_to_uint8(grid))
    save_video_gray(frames, path, fps=fps)

# -------------- main --------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/flow_matching.yaml")
    ap.add_argument("--video-index", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./_global_latent")
    ap.add_argument("--overlap_pct", type=float, default=5.0, help="spatial overlap percent for patchify")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # --- dataset / one val sample ---
    ds_cfg = cfg["val_dataset"]
    DS = try_import_dataset(ds_cfg["name"])
    val_ds = DS(**ds_cfg.get("args", {}))

    sample = val_ds[args.video_index]
    if isinstance(sample, (list, tuple)):
        x_true = sample[0]
    elif isinstance(sample, dict):
        x_true = sample.get("video") or sample.get("x") or sample.get("data")
        if x_true is None:
            raise RuntimeError("Could not find video tensor in dict sample; expected keys 'video' or 'x' or 'data'.")
    else:
        x_true = sample

    x_true = x_true.to(device=device, dtype=torch.float32)  # [2,T,H,W] or [2,1,H,W]
    if x_true.dim() == 3:
        x_true = x_true.unsqueeze(1)
    assert x_true.dim() == 4 and x_true.shape[0] == 2, f"Expected [2,T,H,W], got {tuple(x_true.shape)}"
    _, T_px, H_px, W_px = x_true.shape

    # --- VAE ---
    vae_cfg = cfg.get("vae", {})
    VAE = dynamic_import(vae_cfg["import_path"], vae_cfg["class_name"])
    vae = VAE(**vae_cfg.get("args", {})).to(device).eval()
    vae_ckpt = cfg.get("model", {}).get("vae_ckpt", None) or vae_cfg.get("load_state_dict_from", None)
    if vae_ckpt and os.path.isfile(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
        vae.load_state_dict(state, strict=False)
        print(f"[VAE] loaded weights from {vae_ckpt}")
    for p in vae.parameters():
        p.requires_grad_(False)

    # --- tiling / patch sizes ---
    vcfg   = cfg.get("validation", {})
    ph     = int(vcfg.get("patch_h", 80))
    pw     = int(vcfg.get("patch_w", 80))
    pt     = int(vcfg.get("patch_t", 7))
    bs_vae = int(vcfg.get("patch_batch", 64))
    sh     = pct_to_stride_len(ph, args.overlap_pct)
    sw     = pct_to_stride_len(pw, args.overlap_pct)

    # --- latent patch geometry & scale factors (incl temporal) ---
    with torch.no_grad():
        dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
        vae_out = vae([dummy], op="encode")
        z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
        if z_mu.dim() == 5 and z_mu.shape[0] == 1:
            z_mu = z_mu.squeeze(0)                              # [Cz, nt, Hlp, Wlp]
        Cz, nt, Hlp, Wlp = map(int, (z_mu.shape[0], z_mu.shape[1], z_mu.shape[2], z_mu.shape[3]))

    s_h = Hlp / float(ph)
    s_w = Wlp / float(pw)
    s_t = nt  / float(pt)                                       # <-- temporal compression factor

    H_lat_full = int(math.ceil(H_px * s_h))
    W_lat_full = int(math.ceil(W_px * s_w))
    T_lat_full = int(math.ceil(T_px * s_t))

    print(f"[info] Cz={Cz}, pt={pt} -> nt={nt}  (s_t={s_t:.4f})")
    print(f"[info] latent_patch=({Hlp},{Wlp}), scale≈({s_h:.3f},{s_w:.3f}), global_lat=({T_lat_full},{H_lat_full},{W_lat_full})")

    # --- global latent accumulators (time averaged across overlapping chunks) ---
    Z_num = torch.zeros((Cz, T_lat_full, H_lat_full, W_lat_full), dtype=torch.float32, device=device)
    Z_den = torch.zeros((1, T_lat_full, H_lat_full, W_lat_full), dtype=torch.float32, device=device)

    # --- precompute spatial coords (pixel grid for tiling) ---
    coords_px = spatial_coords(H_px, W_px, ph, pw, sh, sw)
    P = len(coords_px)
    print(f"[tiling] Patches: {P}  (ph,pw)=({ph},{pw})  (sh,sw)=({sh},{sw})")

    # --- iterate pixel-time chunks (1-frame overlap in PIXEL domain) ---
    starts = temporal_chunk_starts(T_px, pt)
    for t0 in tqdm(starts, desc="Global latent build"):
        t1 = min(t0 + pt, T_px)
        Tc_valid = t1 - t0
        x_chunk = x_true[:, t0:t1]                                 # [2, Tc_valid, H, W]
        if Tc_valid < pt:
            pad = pt - Tc_valid
            pad_tail = torch.zeros((2, pad, H_px, W_px), device=device, dtype=x_true.dtype)
            x_chunk = torch.cat([x_chunk, pad_tail], dim=1)        # [2, pt, H, W]

        # spatial patchify -> [P, 2, pt, ph, pw]
        patches_px = spatial_patchify_video(x_chunk, ph, pw, sh, sw, coords_px)

        # encode -> [P, Cz, nt, Hlp, Wlp]
        z_patches = encode_pixel_patches(vae, patches_px, bs_vae)

        # depatchify spatially to a CHUNK latent -> [Cz, nt, H_lat_full, W_lat_full]
        Z_chunk = depatchify_latent_from_pixel_grid(
            z_patches,
            H_px=H_px, W_px=W_px, ph_px=ph, pw_px=pw, sh_px=sh, sw_px=sw,
            coords_px=coords_px
        )

        # temporal placement in LATENT timeline using s_t
        t0_lat = int(round(t0 * s_t))
        t1_lat = t0_lat + nt

        # clip to bounds & adjust usable nt if needed
        if t0_lat >= T_lat_full:
            continue
        if t1_lat > T_lat_full:
            usable = T_lat_full - t0_lat
            if usable <= 0: continue
            Z_chunk = Z_chunk[:, :usable]
            t1_lat = T_lat_full
        else:
            usable = nt

        # simple temporal averaging in overlapped regions
        Z_num[:, t0_lat:t1_lat] += Z_chunk[:, :usable]
        Z_den[:, t0_lat:t1_lat] += 1.0

        # free a bit
        del patches_px, z_patches, Z_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Z_den = torch.where(Z_den == 0, torch.ones_like(Z_den), Z_den)
    Z_global = (Z_num / Z_den).float().detach()                   # [Cz, T_lat, H', W']

    # --- save artifacts ---
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(
        {"latent": Z_global.cpu(), "shape": tuple(Z_global.shape), "s_h": s_h, "s_w": s_w, "s_t": s_t},
        os.path.join(args.outdir, "global_latent.pt")
    )
    fps = int(cfg.get("logging", {}).get("latent_grid_fps", 7))
    save_latent_grid_video(Z_global, os.path.join(args.outdir, "global_latent_grid.mp4"), fps=fps, max_channels=16)

    print("[done] global latent:", tuple(Z_global.shape))
    print("[done] outputs in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
