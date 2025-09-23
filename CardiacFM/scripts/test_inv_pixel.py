#!/usr/bin/env python3
import os, sys, math, time, argparse, importlib, yaml
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

from data.deg import MRIDeg

def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def try_import_dataset(name: str):
    if name in ("CINEDataset", "CINEFlowMatchDataset"):
        mod = importlib.import_module("data.cine_dataset")
        return getattr(mod, "CINEDataset")
    raise ValueError(f"Unknown dataset '{name}'.")

def frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

def save_video_gray(frames_uint8: List[np.ndarray], path: str, fps: int = 7):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio is None:
        base = os.path.splitext(path)[0]
        for i, fr in enumerate(frames_uint8):
            imageio.imwrite(f"{base}_frame_{i:04d}.png", fr)  # type: ignore
        print(f"[warn] imageio-ffmpeg not available. Saved PNG frames under {base}_frame_*.png")
        return
    imageio.mimsave(path, [np.repeat(f[None, ...], 3, axis=0).transpose(1,2,0) for f in frames_uint8],
                    fps=fps, codec="libx264", quality=8)
    print(f"[ok] wrote {path}")

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

def depatchify2d_over_time(patches_P2Thw: torch.Tensor,
                           H: int, W: int, ph: int, pw: int, sh: int, sw: int,
                           coords) -> torch.Tensor:
    device = patches_P2Thw.device
    dtype  = patches_P2Thw.dtype
    P, _, T, _, _ = patches_P2Thw.shape
    out_num = torch.zeros((2, T, H, W), dtype=dtype, device=device)
    out_den = torch.zeros((1, T, H, W), dtype=torch.float32, device=device)

    O1 = max(0, ph - sh); O2 = max(0, pw - sw)
    n1 = coords[0][8]; n2 = coords[0][9]

    for idx, (y0,y1,s1,x0,x1,s2,j,k, *_rest) in enumerate(coords):
        w1 = _axis_weights(s1, j, n1, O1, device)
        w2 = _axis_weights(s2, k, n2, O2, device)
        w = (w1[None, None, :, None] * w2[None, None, None, :])  # [1,1,s1,s2]
        p = patches_P2Thw[idx][:, :, :s1, :s2]
        out_num[:, :, y0:y1, x0:x1] += (p * w).to(out_num.dtype)
        out_den[:, :, y0:y1, x0:x1] += w
    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)

def get_sigmas(cfg, device):
    scfg = cfg.get("sampler", {})
    steps  = int(scfg.get("num_steps", 25))
    shift  = scfg.get("shift", None)
    sigma_k = float(scfg.get("sigma_exponent", 1.0))
    pt = int(cfg.get("validation", {}).get("patch_t", 8))
    ph = int(cfg.get("validation", {}).get("patch_h", 64))
    pw = int(cfg.get("validation", {}).get("patch_w", 64))
    seq_len = int(2 * pt * ph * pw)
    if shift is None:
        mu = calculate_flux_mu(
            seq_len,
            x1=float(scfg.get("x1", 256)), y1=float(scfg.get("y1", 0.5)),
            x2=float(scfg.get("x2", 4096)), y2=float(scfg.get("y2", 1.15)),
            exp_max=float(scfg.get("exp_max", 7.0)),
        )
    else:
        mu = math.log(float(shift))
    sigmas = get_flux_sigmas_from_mu(steps, mu, device=device, dtype=torch.float32)
    if sigma_k != 1.0:
        sigmas = sigmas.pow(sigma_k)
    return sigmas

def flux_time_shift(t: torch.Tensor, mu=1.15, sigma: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = torch.clamp(t, min=1e-5, max=1.0)
    mu_t = torch.as_tensor(mu, device=t.device, dtype=torch.float32)
    emu = torch.exp(mu_t)
    return emu / (emu + (1.0 / t - 1.0).pow(sigma))

def get_flux_sigmas_from_mu(n_steps: int, mu, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t_f32 = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device, dtype=torch.float32)
    return flux_time_shift(t_f32, mu=mu).to(dtype)

def calculate_flux_mu(context_length: int, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0) -> float:
    k = (y2 - y1) / max(1.0, (x2 - x1))
    b = y1 - k * x1
    mu = k * float(context_length) + b
    return float(min(mu, math.log(exp_max)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--video-index", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./_pixel_tests")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & sample
    DS = try_import_dataset(cfg["val_dataset"]["name"])
    val_ds = DS(**cfg["val_dataset"]["args"])
    x_true = val_ds[args.video_index]
    if isinstance(x_true, (list, tuple, dict)):
        raise RuntimeError("Val sample should be a plain [2,T,H,W] tensor in CINEDataset(videos-only).")
    x_true = x_true.to(device=device, dtype=torch.float32)
    if x_true.dim() == 3:
        x_true = x_true.unsqueeze(1)
    _, T, H, W = x_true.shape
    os.makedirs(args.outdir, exist_ok=True)
    gt_frames = [frame_to_uint8(torch.sqrt((x_true[0, t]**2 + x_true[1, t]**2).clamp_min(0)).unsqueeze(0)) for t in range(T)]
    save_video_gray(gt_frames, os.path.join(args.outdir, "gt.mp4"), fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

    # model
    ModelClass = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = ModelClass(**cfg["model"]["args"]).to(device).to(torch.bfloat16).eval()
    for p in model.parameters(): p.requires_grad_(False)
    sd = torch.load(args.ckpt, map_location="cpu")
    state = sd.get("model", sd)
    state = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items() }
    model.load_state_dict(state, strict=False)
    ema_state = sd.get("ema", None)

    # EMA helper
    def with_weights(which: str):
        if which == "ema" and (ema_state is not None):
            model.load_state_dict(ema_state, strict=False)
        elif which == "raw":
            model.load_state_dict(state, strict=False)
        else:
            raise RuntimeError("EMA state not found in ckpt.")

    # shared params
    vcfg = cfg.get("validation", {})
    ph = int(vcfg.get("patch_h", 64)); pw = int(vcfg.get("patch_w", 64))
    pt = int(vcfg.get("patch_t", 8))
    overlap_pct = float(vcfg.get("overlap_spatial_pct", 5.0))
    sh = pct_to_stride_len(ph, overlap_pct); sw = pct_to_stride_len(pw, overlap_pct)
    coords = spatial_coords(H, W, ph, pw, sh, sw)
    P = len(coords)
    sigmas = get_sigmas(cfg, device)
    t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))

    # ---------- unconditional (RAW & EMA) ----------
    from CardiacFM.sampler.flow_match_uni_pc import sample_unipc
    def _append_dims(v, d): return v[(...,) + (None,)*(d - v.ndim)]
    def model_x0(net, x, t):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            u = net(x.to(torch.bfloat16), (t * t_scale).to(torch.bfloat16)).float()
        return x.float() - _append_dims(t, x.ndim) * u

    for which in ["raw", "ema"]:
        with_weights(which)
        N = int(vcfg.get("num_uncond_videos", 8))
        noise = torch.randn(N, 2, pt, ph, pw, device=device, dtype=torch.float32)
        x_samples = sample_unipc(lambda x,t,**_: model_x0(model, x, t),
                                 noise, sigmas, extra_args={}, callback=None, disable=False,
                                 variant=cfg.get("sampler", {}).get("variant", "bh1"))
        # grid
        rows = int(vcfg.get("grid_rows", 2))
        cols = int(vcfg.get("grid_cols", max(1, (N + rows - 1)//rows)))
        frames = []
        for t in range(pt):
            rows_imgs = []
            for r in range(rows):
                cols_imgs = []
                for c in range(cols):
                    idx = r*cols + c
                    if idx < x_samples.shape[0]:
                        patch = x_samples[idx]
                        mag = torch.sqrt(torch.clamp(patch[0, t]**2 + patch[1, t]**2, min=0.0))
                    else:
                        mag = torch.zeros_like(x_samples[0, 0, t])
                    cols_imgs.append(mag)
                rows_imgs.append(torch.cat(cols_imgs, dim=-1))
            grid_img = torch.cat(rows_imgs, dim=-2)
            frames.append(frame_to_uint8(grid_img.unsqueeze(0)))
        save_video_gray(frames, os.path.join(args.outdir, f"uncond_{which}.mp4"),
                        fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

    # ---------- inverse (RAW & EMA): reverse + pix-DC ----------
    def build_zf(x_true_2thw, R):
        deg = MRIDeg(pe=H, fr=T, R=R, dsp=0, verbose=False)
        m_TH = torch.from_numpy(deg.mask_ky_t.T.copy()).to(device=device, dtype=torch.float32)
        m_THW = m_TH[:, :, None].expand(T, H, W)
        xr, xi = x_true_2thw[0], x_true_2thw[1]
        xc = torch.complex(xr, xi)
        k  = torch.fft.fft2(xc, norm="ortho")
        kc = torch.fft.fftshift(k, dim=(-2,-1))
        km = kc * m_THW
        x_zf = torch.fft.ifft2(torch.fft.ifftshift(km, dim=(-2,-1)), norm="ortho")
        return x_zf

    x_zf_full = build_zf(x_true, int(cfg.get("deg", {}).get("R", 8)))  # [T,H,W] complex

    for which in ["raw", "ema"]:
        with_weights(which)
        dc_lambda = float(vcfg.get("dc_lambda", 0.3))
        dc_every  = max(1, int(vcfg.get("dc_every", 4)))
        frames_out = []
        starts = list(range(0, max(1, T - pt + 1), max(1, pt - 1)))
        if starts and starts[-1] != T - pt: starts.append(T - pt)

        for t0 in starts:
            t1 = min(t0 + pt, T)
            Tc_valid = t1 - t0
            x_zf_chunk = x_zf_full[t0:t1]   # [Tc_valid,H,W] complex
            if Tc_valid < pt:
                pad = pt - Tc_valid
                x_zf_chunk = torch.cat([x_zf_chunk,
                                        torch.zeros((pad, H, W), device=device, dtype=x_zf_chunk.dtype)], dim=0)

            noise_full = torch.randn((2, pt, H, W), device=device, dtype=torch.float32)
            x = spatial_patchify_video(noise_full, ph, pw, sh, sw, coords).float()  # [P,2,pt,ph,pw]
            B = x.shape[0]; total = sigmas.numel() - 1

            for i in tqdm(range(total), desc=f"inv[{which}] t0={t0}", leave=False):
                t = sigmas[i].expand(B); s = sigmas[i+1].expand(B)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    u = model(x.to(torch.bfloat16), (t * t_scale).to(torch.bfloat16)).float()
                x0_pred = x - (t[(...,) + (None,)*(x.ndim-1)]) * u

                do_dc = (i % dc_every == 0) or (i == total - 1)
                if do_dc:
                    x0_pred = x0_pred.detach().requires_grad_(True)
                    # depatchify
                    x_full = depatchify2d_over_time(x0_pred, H, W, ph, pw, sh, sw, coords)  # [2,pt,H,W]
                    xr, xi = x_full[0], x_full[1]
                    x_pred_c = torch.complex(xr, xi)[:Tc_valid]
                    diff = x_pred_c - x_zf_chunk[:Tc_valid]
                    L = (diff.real.pow(2) + diff.imag.pow(2)).mean()
                    (g,) = torch.autograd.grad(L, x0_pred, retain_graph=False, create_graph=False)
                    step_sz = (t - s).abs() * dc_lambda
                    x0_pred = (x0_pred - step_sz[(...,) + (None,)*(x.ndim-1)] * g).detach()

                ratio = (s / t.clamp_min(1e-8))[(...,) + (None,)*(x.ndim-1)]
                x = ratio * x + (1.0 - ratio) * x0_pred

            x_full = depatchify2d_over_time(x, H, W, ph, pw, sh, sw, coords)  # [2,pt,H,W]
            for f in range(Tc_valid):
                frames_out.append(frame_to_uint8(torch.sqrt((x_full[0, f]**2 + x_full[1, f]**2).clamp_min(0)).unsqueeze(0)))

        save_video_gray(frames_out, os.path.join(args.outdir, f"inverse_{which}.mp4"),
                        fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

if __name__ == "__main__":
    main()
