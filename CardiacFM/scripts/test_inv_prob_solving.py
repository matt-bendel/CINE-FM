#!/usr/bin/env python3
import os, sys, math, time, argparse, importlib, yaml
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

def load_teacache_poly(json_path: str):
    import json, os
    if (json_path is None) or (not os.path.isfile(json_path)):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    # expects keys: 'coeffs_high_to_low' and 'poly_order'
    return data.get("coeffs_high_to_low", None)

# ---------- utils: dynamic import & dataset ----------
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

def ragged_collate(batch):
    return batch

# ---------- tiny viz ----------
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
        print(f"[warn] imageio-ffmpeg not available. Saved PNG frames to {base}_frame_*.png")
        return
    try:
        imageio.mimsave(path, [np.repeat(f[None, ...], 3, axis=0).transpose(1,2,0) for f in frames_uint8],
                        fps=fps, codec="libx264", quality=8)
        print(f"[ok] wrote {path}")
    except Exception as e:
        gif_path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, frames_uint8, fps=fps)
        print(f"[ok] wrote {gif_path} (mp4 failed: {e})")

# ---------- patchify / depatchify (spatial-only over time) ----------
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
        p = patches_P2Thw[idx][:, :, :s1, :s2]                   # [2,T,s1,s2]
        out_num[:, :, y0:y1, x0:x1] += (p * w).to(out_num.dtype)
        out_den[:, :, y0:y1, x0:x1] += w
    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)

def temporal_chunk_starts(T: int, pt: int) -> List[int]:
    if pt >= T: return [0]
    step = max(1, pt - 1)  # 1-frame overlap
    starts = list(range(0, T - pt + 1, step))
    if starts[-1] != T - pt:
        starts.append(T - pt)
    return starts

# ---------- VAE encode/decode helpers ----------
@torch.no_grad()
def decode_latent_patches(vae, z_Pcnhw: torch.Tensor, bs_vae: int) -> torch.Tensor:
    outs = []
    for i in range(0, z_Pcnhw.shape[0], bs_vae):
        chunk = z_Pcnhw[i:i+bs_vae]
        z_list = [z.unsqueeze(0) for z in chunk]         # [1,Cz,nt,H',W']
        dec = vae(z_list, op="decode")                   # list of [1,2,pt,ph,pw]
        for o in dec:
            if isinstance(o, (list, tuple)):
                o = o[0]
            outs.append(o.squeeze(0))                    # [2,pt,ph,pw]
    return torch.stack(outs, dim=0)

@torch.no_grad()
def encode_pixel_patches(vae, patches_P2thw: torch.Tensor, bs_vae: int) -> torch.Tensor:
    outs = []
    for i in range(0, patches_P2thw.shape[0], bs_vae):
        chunk = patches_P2thw[i:i+bs_vae]
        x_list = [x for x in chunk]                      # list of [2,pt,ph,pw]
        pairs = vae(x_list, op="encode")                 # list of (mu, logv)
        for pr in pairs:
            mu = pr[0] if isinstance(pr, (list, tuple)) else pr
            if mu.dim() == 5 and mu.shape[0] == 1:
                mu = mu.squeeze(0)
            outs.append(mu)                              # [Cz,nt,H',W']
    return torch.stack(outs, dim=0)

# ---------- core: run one full-video inverse validation for a given model ----------
def run_validation_inv_once(
    model,
    vae,
    x_true_2thw: torch.Tensor,
    cfg: Dict[str, Any],
    use_linear_sigmas: bool = True,
    t_scale: float = 1000.0,
    R: int = 8,
    outdir: str = ".",
    tag: str = "raw",
    save_chunk_videos: bool = False,
    seed: int = 1234,
    # DC knobs (interpreted for grad-based DC here)
    phase_only: bool = True,       # (kept for API; not used in pixel-space DC below)
    dc_lambda: float = 0.3,        # scales per-step DC step size (see eta_dc below)
    dc_every: int = 4,             # apply DC every N solver steps (and on last)
    save_center_over_steps: bool = True,
    save_full_every: int = 0,
):
    """
    Euler(x0) + periodic **pixel-space** data consistency with gradients:

      • Measurements used by DC are the zero-filled pixels x_zf (complex).
      • For a candidate latent x0 (patch-batched), we:
          1) decode → depatchify to pixels,
          2) undersample in k-space with the same centered mask, IFFT back to pixels,
          3) compute L2 loss vs. x_zf (pixel-space),
          4) autograd to get ∂L/∂(latent patches) for each patch, and
          5) take a small step in latent space.

      The step size per solver step is:  eta_dc = dc_lambda * |t - s|.
      DC is applied every `dc_every` steps and at the last step.
    """
    device = x_true_2thw.device
    vcfg   = cfg.get("validation", {})
    ph     = int(vcfg.get("patch_h", 80))
    pw     = int(vcfg.get("patch_w", 80))
    pt     = int(vcfg.get("patch_t", 7))
    bs_vae = int(vcfg.get("patch_batch", 64))
    overlap_spatial_pct = 1.0
    dc_lambda = float(max(0.0, min(1.0, dc_lambda)))
    dc_every  = max(1, int(dc_every))

    # ---------- spatial tiling ----------
    def pct_to_stride_len(P: int, pct: float) -> int:
        ov = max(0.0, min(99.0, float(pct))) / 100.0
        return max(1, int(math.ceil(P * (1.0 - ov))))
    sh = pct_to_stride_len(ph, overlap_spatial_pct)
    sw = pct_to_stride_len(pw, overlap_spatial_pct)
    _, T, H, W = x_true_2thw.shape
    coords = spatial_coords(H, W, ph, pw, sh, sw)
    P = len(coords)

    # ---------- latent geometry ----------
    with torch.no_grad():
        dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
        vae_out = vae([dummy], op="encode")
        mu0 = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
        if mu0.dim() == 5 and mu0.shape[0] == 1:
            mu0 = mu0.squeeze(0)
        Cz, nt, Hlat, Wlat = int(mu0.shape[0]), int(mu0.shape[1]), int(mu0.shape[2]), int(mu0.shape[3])

    # ---------- schedule ----------
    steps = int(cfg.get("sampler", {}).get("num_steps", 25))
    sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

    # ---------- measured (centered) mask + zero-filled pixels ----------
    from data.deg import MRIDeg
    deg_full = MRIDeg(pe=H, fr=T, R=R, dsp=0, verbose=False)
    m_full_TH = torch.from_numpy(deg_full.mask_ky_t.T.copy()).to(device=device, dtype=torch.float32)  # [T,H]
    m_full_THW = m_full_TH[:, :, None].expand(T, H, W)                                                # [T,H,W]

    xr, xi = x_true_2thw[0], x_true_2thw[1]
    xc_true = torch.complex(xr, xi)                                       # [T,H,W]
    k_true  = torch.fft.fft2(xc_true, norm="ortho")
    k_true_c = torch.fft.fftshift(k_true, dim=(-2, -1))
    k_meas_c_full = k_true_c * m_full_THW                                 # [T,H,W] complex

    # zero-filled (complex pixel space) reference
    x_zf_full = torch.fft.ifft2(torch.fft.ifftshift(k_meas_c_full, dim=(-2, -1)), norm="ortho")  # [T,H,W] complex

    # Save zero-filled magnitude video (optional but handy)
    x_zf_ri = torch.stack((x_zf_full.real, x_zf_full.imag), dim=0)
    zf_frames = []
    for t in range(T):
        mag = torch.sqrt(torch.clamp(x_zf_ri[0, t]**2 + x_zf_ri[1, t]**2, min=0.0)).unsqueeze(0)
        zf_frames.append(frame_to_uint8(mag))
    save_video_gray(zf_frames, os.path.join(outdir, f"zfr_{tag}.mp4"),
                    fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

    # ---------- helpers ----------
    def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
        return v[(...,) + (None,) * (target_ndim - v.ndim)]

    def depatchify(patches_P2Thw: torch.Tensor) -> torch.Tensor:
        return depatchify2d_over_time(patches_P2Thw, H, W, ph, pw, sh, sw, coords)

    # grad-enabled local decode (the global helper is @torch.no_grad(); we need grads)
    def _decode_latent_patches_grad(vae, z_Pcnhw: torch.Tensor, bs: int) -> torch.Tensor:
        outs = []
        for i in range(0, z_Pcnhw.shape[0], bs):
            chunk = z_Pcnhw[i:i+bs]
            z_list = [zz.unsqueeze(0) for zz in chunk]       # [1,Cz,nt,H',W']
            dec = vae(z_list, op="decode")                   # list of [1,2,pt,ph,pw]
            for o in dec:
                if isinstance(o, (list, tuple)): o = o[0]
                outs.append(o.squeeze(0))                    # [2,pt,ph,pw]
        return torch.stack(outs, dim=0)                      # [P,2,pt,ph,pw]
    
    def _decode_latent_patches_grad_streamed(vae, z_Pcnhw: torch.Tensor, max_bs: int = 1):
        """
        Decode with gradients but in tiny microbatches to cap peak memory.
        Input:  z [P, Cz, nt, H', W']
        Return: x [P, 2, pt, ph, pw] (requires grad wrt z_Pcnhw)
        """
        P = int(z_Pcnhw.shape[0])
        outs = []
        for i in range(0, P, max_bs):
            z_chunk = z_Pcnhw[i:i+max_bs]                     # [B, Cz, nt, H', W']
            # VAE API expects a list of [1,Cz,nt,H',W'] tensors
            z_list = [zz.unsqueeze(0) for zz in z_chunk]      # list length B
            # Important: do NOT use torch.no_grad, and keep dtype fp32/bf16 as you prefer
            dec_list = vae(z_list, op="decode")               # list of [1,2,pt,ph,pw]
            for o in dec_list:
                if isinstance(o, (list, tuple)):
                    o = o[0]
                outs.append(o.squeeze(0))                     # [2,pt,ph,pw], grad-connected to z_chunk
            # free cached blocks between microbatches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.stack(outs, dim=0)                       # [P,2,pt,ph,pw]


    # ---------- quick VAE round-trip sanity (warn if large) ----------
    with torch.no_grad():
        z_test = torch.randn(4, Cz, nt, Hlat, Wlat, device=device, dtype=torch.float32)
        x_test = decode_latent_patches(vae, z_test, bs_vae)             # [4,2,pt,ph,pw]
        z_back = encode_pixel_patches(vae, x_test, bs_vae)              # [4,Cz,nt,H',W']
        rt_rel = (z_back - z_test).pow(2).mean().sqrt() / (z_test.pow(2).mean().sqrt() + 1e-8)
        if float(rt_rel) > 0.15:
            print(f"[warn] VAE encode↔decode relative RMSE ~ {float(rt_rel):.3f} (possible VAE mismatch).")

    # ---------- pixel-space DC (with grad) ----------
    def dc_latent_from_x0(
        x0_lat_Pcnhw: torch.Tensor,
        x_zf_TcHW: torch.Tensor,      # zero-filled target in pixel space [Tc,H,W] (complex)
        m_chunk_TcHW: torch.Tensor,   # [Tc,H,W] (unused in pixel-space loss; keep if needed elsewhere)
        step_size: torch.Tensor,      # per-batch step size (broadcastable)
    ):
        # ---- ensure z (latents) require grad ----
        z = x0_lat_Pcnhw
        if not z.requires_grad:
            z = z.clone().requires_grad_(True)

        print(z.shape)
        # ---- decode (streamed, grad-enabled) ----
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x_patches = _decode_latent_patches_grad_streamed(vae, z.to(torch.float32), max_bs=1)   # [P,2,pt,ph,pw]
        
        x_patches = x_patches.to(torch.float32)

        # ---- depatchify to full chunk pixels (keeps grad) ----
        x_full = depatchify(x_patches)       # [2, pt, H, W]; differentiable wrt z
        # Align Tc if needed (as in your original)
        _, pt_eff, H_eff, W_eff = x_full.shape
        Tc = x_zf_TcHW.shape[0]
        if pt_eff != Tc:
            if Tc < pt_eff:
                pad = pt_eff - Tc
                x_zf = torch.cat([
                    x_zf_TcHW,
                    torch.zeros((pad, H_eff, W_eff), device=z.device, dtype=x_zf_TcHW.dtype)
                ], dim=0)
            else:
                x_zf = x_zf_TcHW[:pt_eff]
        else:
            x_zf = x_zf_TcHW

        # ---- build predicted complex from real/imag ----
        xr, xi = x_full[0], x_full[1]                         # [pt, H, W]
        x_pred_c = torch.complex(xr, xi)                      # [pt, H, W]

        # ---- LOSS in PIXEL space: || x_pred - x_zf ||_2^2 (masked ZF already) ----
        # x_zf is complex [pt,H,W]
        diff = x_pred_c - x_zf
        L = (diff.real.pow(2) + diff.imag.pow(2)).mean()      # scalar; requires grad

        # ---- grad wrt z ----
        (g,) = torch.autograd.grad(L, z, retain_graph=False, create_graph=False)  # [P,Cz,nt,H',W']

        # ---- gradient step in latent space ----
        # step_size shape: broadcastable to z
        z_new = z - step_size[(...,) + (None,)*(z.ndim - step_size.ndim)] * g
        return z_new

    # ---------- Euler(x0) with periodic DC ----------
    def sample_euler_x0_with_dc(
        net,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
        x_zf_full: torch.Tensor,     # [T,H,W] complex (pixel-space)
        m_full: torch.Tensor,        # [T,H,W] float (centered k-space mask)
        t_start: int,
        Tc_valid: int,
        save_dir_for_chunk: str,
    ) -> torch.Tensor:

        x = noise.to(torch.float32)
        B = x.shape[0]
        total = sigmas.numel() - 1

        # TeaCache init per chunk
        tc_cfg = cfg.get("teacache", {})
        tc_enable = bool(tc_cfg.get("enable", False))
        tc_thresh = float(tc_cfg.get("rel_l1_thresh", 0.15))
        tc_poly_file = tc_cfg.get("poly_file", None)
        tc_coeffs = load_teacache_poly(tc_poly_file) if tc_enable else None
        if tc_enable:
            net.initialize_teacache(
                enable_teacache=True,
                num_steps=int(total),
                rel_l1_thresh=tc_thresh,
                poly_coeffs=tc_coeffs
            )
        else:
            net.initialize_teacache(enable_teacache=False)

        # slice measurements/masks for this chunk (pad tail to pt)
        t_end_valid = min(t_start + Tc_valid, T)
        x_zf_valid = x_zf_full[t_start:t_end_valid]           # [Tc_valid,H,W] complex
        m_valid     = m_full[t_start:t_end_valid]             # [Tc_valid,H,W] float
        if Tc_valid < pt:
            pad_c = torch.zeros((pt - Tc_valid, H, W), device=device, dtype=x_zf_valid.dtype)
            pad_m = torch.zeros((pt - Tc_valid, H, W), device=device, dtype=m_valid.dtype)
            x_zf_chunk = torch.cat([x_zf_valid, pad_c], dim=0)   # [pt,H,W] complex
            m_chunk    = torch.cat([m_valid,     pad_m], dim=0)  # [pt,H,W] float
        else:
            x_zf_chunk = x_zf_valid
            m_chunk    = m_valid

        center_mag_frames: List[np.ndarray] = []
        center_idx = min(max(Tc_valid // 2, 0), pt - 1)
        do_full = (save_full_every is not None) and (int(save_full_every) > 0)
        full_every = int(save_full_every) if do_full else 0
        dc_counter = 0

        for i in tqdm(range(total)):
            t = sigmas[i].expand(B)      # fp32
            s = sigmas[i + 1].expand(B)

            # model velocity (no grads wrt model)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    u = net(x.to(torch.bfloat16), (t * float(t_scale)).to(torch.bfloat16))
                u = u.to(torch.float32)

            # Euler(x0)-style predictor
            x0_g_t = x - _append_dims(t, x.ndim) * u
            x1_g_t = x + (1 - _append_dims(t, x.ndim)) * u

            # --- periodic pixel-space DC (with grads wrt latent) ---
            # per-patch step size: scale by |t - s| and dc_lambda
            # t,s are shape [B] where B==P (patch batch)
            step_sz = (t - s).abs() * dc_lambda         # [P]
            x0_g_y = dc_latent_from_x0(x0_g_t, x_zf_chunk, m_chunk, step_sz)

            # (Optional logging: center magnitude frame over steps)
            if save_center_over_steps:
                with torch.no_grad():
                    # decode -> depatchify (no grad needed here)
                    px = decode_latent_patches(vae, x0_g_t, bs_vae)       # [P,2,pt,ph,pw]
                    xf = depatchify(px)                                    # [2,pt,H,W]
                    mag = torch.sqrt(torch.clamp(xf[0, center_idx]**2 + xf[1, center_idx]**2, min=0.0)).unsqueeze(0)
                    center_mag_frames.append(frame_to_uint8(mag))

            # combine (same as your earlier xt0/xt1 mix)
            xt0_g_t = _append_dims(t, x.ndim) * x0_g_y + (1 - _append_dims(t, x.ndim)) * x0_g_t
            xt1_g_t = torch.sqrt(_append_dims(s, x.ndim)) * x1_g_t + torch.sqrt((1 - _append_dims(s, x.ndim))) * torch.randn_like(x)
            x = (1 - _append_dims(s, x.ndim)) * xt0_g_t + _append_dims(s, x.ndim) * xt1_g_t
            x = x.detach()  # avoid graph growth across steps

        if save_center_over_steps and len(center_mag_frames) > 0:
            save_video_gray(center_mag_frames,
                            os.path.join(save_dir_for_chunk, f"chunk_t{t_start:04d}_center_over_steps.mp4"),
                            fps=10)
        return x

    # ---------- deterministic noise ----------
    g = torch.Generator(device=device); g.manual_seed(int(seed))

    # ---------- reconstruct whole video ----------
    recon_frames: List[torch.Tensor] = []
    starts = temporal_chunk_starts(T, pt)
    prev_start, prev_len = None, None

    inter_dir = os.path.join(outdir, f"intermediate_{tag}")
    if save_center_over_steps or (save_full_every and save_full_every > 0):
        os.makedirs(inter_dir, exist_ok=True)

    for t0 in starts:
        t1 = min(t0 + pt, T)
        Tc_valid = int(t1 - t0)

        noise = torch.randn((P, Cz, nt, Hlat, Wlat), generator=g, device=device, dtype=torch.float32)
        z_x0 = sample_euler_x0_with_dc(
            model, noise, sigmas,
            x_zf_full=x_zf_full, m_full=m_full_THW,
            t_start=t0, Tc_valid=Tc_valid,
            save_dir_for_chunk=inter_dir,
        )  # [P,Cz,nt,Hlat,Wlat]

        patches_dec = decode_latent_patches(vae, z_x0, bs_vae)             # [P,2,pt,ph,pw]
        xhat_full   = depatchify(patches_dec)                              # [2,pt,H,W]
        xhat_valid  = xhat_full[:, :Tc_valid]

        if prev_start is None:
            for f in range(Tc_valid):
                recon_frames.append(xhat_valid[:, f])
            prev_start, prev_len = t0, Tc_valid
        else:
            overlap = max(0, (prev_start + prev_len) - t0)
            start_f = min(overlap, Tc_valid)
            for f in range(start_f, Tc_valid):
                recon_frames.append(xhat_valid[:, f])
            prev_start, prev_len = t0, Tc_valid

        if save_chunk_videos:
            frames_u8 = []
            for f in range(Tc_valid):
                mag = torch.sqrt(torch.clamp(xhat_valid[0, f]**2 + xhat_valid[1, f]**2, min=0.0)).unsqueeze(0)
                frames_u8.append(frame_to_uint8(mag))
            save_video_gray(frames_u8, os.path.join(outdir, f"chunk_{tag}_{t0:04d}.mp4"),
                            fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recon = torch.stack(recon_frames, dim=1)                               # [2,T,H,W]
    assert recon.shape[1] == T, f"Reconstruction length {recon.shape[1]} != GT {T}"

    frames_u8 = []
    for f in range(T):
        mag = torch.sqrt(torch.clamp(recon[0, f]**2 + recon[1, f]**2, min=0.0)).unsqueeze(0)
        frames_u8.append(frame_to_uint8(mag))
    save_video_gray(frames_u8, os.path.join(outdir, f"recon_{tag}.mp4"),
                    fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))
    return recon

@torch.no_grad()
def run_uncond_debug_once(
    model,
    vae,
    x_ref_2thw: torch.Tensor,          # only used to get (H,W)
    cfg: Dict[str, Any],
    outdir: str,
    tag: str = "uncond_debug",
    seed: int = 1234,
) -> torch.Tensor:
    """
    Unconditional debug sampler:
      • tiles the spatial field of view using (ph,pw) with 5% overlap
      • samples P independent latent videos via Euler(x0) **without autocast**
      • decodes and depatchifies to a [2, pt, H, W] video
      • saves magnitude MP4 to {outdir}/uncond_{tag}.mp4
    """
    device = next(model.parameters()).device
    vcfg   = cfg.get("validation", {})
    ph     = int(vcfg.get("patch_h", 80))
    pw     = int(vcfg.get("patch_w", 80))
    pt     = int(vcfg.get("patch_t", 7))
    bs_vae = int(vcfg.get("patch_batch", 64))

    t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))
    steps   = int(cfg.get("sampler", {}).get("num_steps", 25))

    # --- field of view (just use the reference tensor's H,W) ---
    _, _, H, W = x_ref_2thw.shape

    # --- spatial tiling ---
    sh = pct_to_stride_len(ph, 5.0)
    sw = pct_to_stride_len(pw, 5.0)
    coords = spatial_coords(H, W, ph, pw, sh, sw)
    P = len(coords)

    # --- latent geometry via dummy encode ---
    dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
    vae_out = vae([dummy], op="encode")
    z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
    if z_mu.dim() == 5 and z_mu.shape[0] == 1:
        z_mu = z_mu.squeeze(0)  # [Cz, nt, H', W']
    Cz, nt, Hlat, Wlat = int(z_mu.shape[0]), int(z_mu.shape[1]), int(z_mu.shape[2]), int(z_mu.shape[3])

    # --- linear schedule 1→0 (inclusive), fp32 state ---
    sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

    tc_cfg = cfg.get("teacache", {})
    tc_enable = bool(tc_cfg.get("enable", False))
    tc_thresh = float(tc_cfg.get("rel_l1_thresh", 0.15))
    tc_poly_file = tc_cfg.get("poly_file", None)
    tc_coeffs = _load_teacache_poly(tc_poly_file) if tc_enable else None

    if tc_enable:
        model.initialize_teacache(
            enable_teacache=True,
            num_steps=int(steps),
            rel_l1_thresh=tc_thresh,
            poly_coeffs=tc_coeffs
        )
    else:
        model.initialize_teacache(enable_teacache=False)

    # quick sanity prints (match training)
    if P == 0:
        raise RuntimeError("No spatial tiles computed.")
    print(f"[uncond] Cz={Cz} nt={nt} H'={Hlat} W'={Wlat}  P={P}  steps={steps}  t_scale={t_scale}")

    def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
        return v[(...,) + (None,) * (target_ndim - v.ndim)]

    # --- Euler(x0) sampler (IDENTICAL logic to training's validation_simple) ---
    def _sample_euler_x0_velocity(net, noise, sigmas, t_scale=1000.0):
        x = noise.float()
        B = x.shape[0]
        total = sigmas.numel() - 1
        for i in tqdm(range(total), desc="Solving"):
            t = sigmas[i].expand(B)      # fp32
            s = sigmas[i+1].expand(B)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                u = net(x.to(torch.bfloat16), (t * t_scale).to(torch.bfloat16))
            u = u.float()

            x0_pred = x - t[(...,) + (None,)*(x.ndim-1)] * u
            ratio   = (s / t.clamp_min(1e-8))[(...,) + (None,)*(x.ndim-1)]
            x = ratio * x + (1.0 - ratio) * x0_pred
        return x


    # --- unconditional sampling per spatial patch ---
    g = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((P, Cz, nt, Hlat, Wlat), generator=g, device=device, dtype=torch.float32)
    z_x0  = _sample_euler_x0_velocity(model, noise, sigmas)                 # [P,Cz,nt,H',W']

    # --- decode to pixel patches & depatchify to full [2, pt, H, W] ---
    patches_dec = decode_latent_patches(vae, z_x0, bs_vae)                   # [P,2,pt,ph,pw]
    xhat_full   = depatchify2d_over_time(patches_dec, H, W, ph, pw, sh, sw, coords)  # [2,pt,H,W]

    # --- save magnitude video ---
    frames_u8 = []
    for t in range(int(xhat_full.shape[1])):
        mag = torch.sqrt(torch.clamp(xhat_full[0, t]**2 + xhat_full[1, t]**2, min=0.0)).unsqueeze(0)
        frames_u8.append(frame_to_uint8(mag))
    os.makedirs(outdir, exist_ok=True)
    save_video_gray(frames_u8, os.path.join(outdir, f"uncond_{tag}.mp4"),
                    fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))
    return xhat_full  # [2, pt, H, W]


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default='configs/flow_matching.yaml')
    ap.add_argument("--ckpt", type=str, default='/storage/matt_models/latent_fm/flowmatch/step_0380000/state.pt', help="FM checkpoint with model/ema states")
    ap.add_argument("--use-ema", type=str, default="raw", choices=["raw", "ema", "both"],
                    help="Which weights to test")
    ap.add_argument("--video-index", type=int, default=0, help="Deterministic val sample index")
    ap.add_argument("--R", type=int, default=8, help="Acceleration factor")
    ap.add_argument("--outdir", type=str, default="./_test_inv_out")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build val dataset & fetch ONE item deterministically ---
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

    x_true = x_true.to(device=device, dtype=torch.float32)
    if x_true.dim() == 3:
        x_true = x_true.unsqueeze(1)  # [2,1,H,W]
    assert x_true.dim() == 4 and x_true.shape[0] == 2, f"Expected [2,T,H,W], got {tuple(x_true.shape)}"

    # save GT magnitude video
    os.makedirs(args.outdir, exist_ok=True)
    _, T, H, W = x_true.shape
    gt_frames = []
    for t in range(T):
        mag = torch.sqrt(torch.clamp(x_true[0, t]**2 + x_true[1, t]**2, min=0.0)).unsqueeze(0)
        gt_frames.append(frame_to_uint8(mag))
    save_video_gray(gt_frames, os.path.join(args.outdir, "gt.mp4"),
                    fps=int(cfg.get("logging", {}).get("latent_grid_fps", 7)))

    # --- build VAE ---
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

    # --- build models (raw / ema) ---
    ModelClass = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])

    def load_model_from_ckpt(which: str, vae):
        m = ModelClass(**cfg["model"]["args"]).to(device).eval()

        # --- infer latent geometry from VAE using your validation patch sizes ---
        vcfg = cfg.get("validation", {})
        pt = int(vcfg.get("patch_t", 7))
        ph = int(vcfg.get("patch_h", 80))
        pw = int(vcfg.get("patch_w", 80))

        with torch.no_grad():
            dummy_px = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
            vae_out = vae([dummy_px], op="encode")
            z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
            if z_mu.dim() == 5 and z_mu.shape[0] == 1:
                z_mu = z_mu.squeeze(0)
            Cz, nt, Hlat, Wlat = int(z_mu.shape[0]), int(z_mu.shape[1]), int(z_mu.shape[2]), int(z_mu.shape[3])

        # --- now strict load is safe (final.* exists) ---
        ckpt_path = args.ckpt or cfg["model"].get("load_state_dict_from", None)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError(f"bad checkpoint path: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        state = sd.get("model")
        if state is None:
            raise RuntimeError("checkpoint has no 'model' state_dict")

        # strip potential wrappers
        state = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items() }

        # strict=True now that shapes match (incl. final.*)
        m.load_state_dict(state, strict=True)

        # freeze & set dtype for inference
        m.to(torch.bfloat16).eval()
        for p in m.parameters():
            p.requires_grad_(False)

        m = torch.compile(m)
        return m

    do_raw = args.use_ema in ("raw", "both")
    do_ema = args.use_ema in ("ema", "both")

    model_raw = load_model_from_ckpt("raw", vae)
    # _ = run_uncond_debug_once(model_raw, vae, x_true, cfg, outdir=args.outdir, tag="raw")
    recon_raw = run_validation_inv_once(
        model_raw, vae, x_true, cfg,
        use_linear_sigmas=True, t_scale=float(cfg.get("sampler", {}).get("t_scale", 1000.0)),
        R=int(args.R), outdir=args.outdir, tag="raw",
        save_chunk_videos=False, seed=int(args.seed)
    )
    del model_raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    print("[done] outputs in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
