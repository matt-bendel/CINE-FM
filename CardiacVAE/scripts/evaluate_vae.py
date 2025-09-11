#!/usr/bin/env python3
import os, sys, json, math, random, argparse
from math import ceil
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- project imports ---
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

from data.cine_dataset import CINEDataset  # returns [2,1,H,W] (pretrain_2d) or [2,L,H,W] (videos)

# ======================= Patch helpers (mirrors training) =======================
def compute_stride_and_n_patches(D, P, extra_patch_num=0):
    n_patches = 1
    while True:
        if n_patches == 1:
            S = 1
        else:
            S = ceil((D - P) / (n_patches - 1))
            S = max(S, 1)
        last_patch_end = (n_patches - 1) * S + P
        if last_patch_end >= D:
            starts = [i * S for i in range(n_patches)]
            coverage = set()
            for start in starts:
                end = start + P
                coverage.update(range(start, min(end, D)))
            if len(coverage) >= D:
                break
        n_patches += 1
    n_patches += extra_patch_num
    if n_patches > 1:
        S = ceil((D - P) / (n_patches - 1))
        S = max(S, 1)
    return S, n_patches

def compute_strides_and_N(data_shape, patch_size=(80, 80, 11), extra_patch_num=(0,0,0)):
    strides = []
    n_patches_list = []
    for D, P, extra in zip(data_shape, patch_size, extra_patch_num):
        S, n_patches = compute_stride_and_n_patches(D, P, extra)
        strides.append(S)
        n_patches_list.append(n_patches)
    N = torch.prod(torch.tensor(n_patches_list))
    return tuple(strides), N.item(), tuple(n_patches_list)

def patchify(data: torch.Tensor, patch_size):
    data_shape = data.shape
    strides, N, n_patches_list = compute_strides_and_N(data_shape, patch_size)
    S0, S1, S2 = strides
    P0, P1, P2 = patch_size
    n_patches_d0, n_patches_d1, n_patches_d2 = n_patches_list

    patches = torch.zeros((N, P0, P1, P2), dtype=data.dtype, device=data.device)
    patch_idx = 0

    for i in range(n_patches_d0):
        start0 = i * S0
        end0 = start0 + P0
        actual_end0 = min(end0, data_shape[0])
        for j in range(n_patches_d1):
            start1 = j * S1
            end1 = start1 + P1
            actual_end1 = min(end1, data_shape[1])
            for k in range(n_patches_d2):
                start2 = k * S2
                end2 = start2 + P2
                actual_end2 = min(end2, data_shape[2])
                patch = data[start0:actual_end0, start1:actual_end1, start2:actual_end2]
                padded_patch = torch.zeros((P0, P1, P2), dtype=data.dtype, device=data.device)
                s0 = actual_end0 - start0
                s1 = actual_end1 - start1
                s2 = actual_end2 - start2
                padded_patch[:s0, :s1, :s2] = patch
                patches[patch_idx] = padded_patch
                patch_idx += 1

    return patches, strides

def depatchify(patches: torch.Tensor, data_shape: tuple, patch_size: tuple, strides: tuple):
    D0, D1, D2 = data_shape
    P0, P1, P2 = patch_size
    S0, S1, S2 = strides

    device = patches.device
    dtype  = patches.dtype

    out_num = torch.zeros(data_shape, dtype=dtype, device=device)
    out_den = torch.zeros(data_shape, dtype=torch.float32, device=device)

    n0 = max(1, ceil((D0 - P0) / S0) + 1)
    n1 = max(1, ceil((D1 - P1) / S1) + 1)
    n2 = max(1, ceil((D2 - P2) / S2) + 1)

    expected_N = n0 * n1 * n2
    if patches.shape[0] != expected_N:
        raise ValueError(f"patches.shape[0]={patches.shape[0]} != expected {expected_N} (= {n0}*{n1}*{n2})")

    O0 = max(0, P0 - S0); O1 = max(0, P1 - S1); O2 = max(0, P2 - S2)

    def axis_weights(L_eff: int, idx: int, n: int, O: int) -> torch.Tensor:
        has_prev, has_next = (idx > 0), (idx < n - 1)
        L_left  = min(O if has_prev else 0, L_eff)
        L_right = min(O if has_next else 0, L_eff)
        if L_left + L_right > L_eff:
            if L_left > 0 and L_right > 0:
                total = L_left + L_right
                L_left_new  = max(1, int(round(L_eff * (L_left / total))))
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

    patch_idx = 0
    for i in range(n0):
        start0 = i * S0; end0 = min(start0 + P0, D0)
        s0, ps0, w0 = slice(start0, end0), slice(0, end0 - start0), axis_weights(end0 - start0, i, n0, O0)
        for j in range(n1):
            start1 = j * S1; end1 = min(start1 + P1, D1)
            s1, ps1, w1 = slice(start1, end1), slice(0, end1 - start1), axis_weights(end1 - start1, j, n1, O1)
            for k in range(n2):
                start2 = k * S2; end2 = min(start2 + P2, D2)
                s2, ps2, w2 = slice(start2, end2), slice(0, end2 - start2), axis_weights(end2 - start2, k, n2, O2)
                w = (w0[:, None, None] * w1[None, :, None] * w2[None, None, :])
                patch = patches[patch_idx][ps0, ps1, ps2]
                out_num[s0, s1, s2] += patch * w.to(dtype)
                out_den[s0, s1, s2] += w
                patch_idx += 1

    out_den[out_den == 0] = 1.0
    return out_num / out_den.to(dtype)

def ragged_collate(batch):  # keep ragged batch (list of tensors)
    return batch

# ======================= Complex & viz helpers =======================
@torch.no_grad()
def complex_mag(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:  # [2,T,H,W] -> [1,T,H,W]
    return (x.pow(2).sum(dim=0, keepdim=True) + eps).sqrt()

@torch.no_grad()
def snr_complex(x: torch.Tensor, xhat: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Complex SNR in dB on the full complex tensor [2,T,H,W]:
        SNR(xhat,x) = 10 * log10( ||x||^2 / ||xhat - x||^2 )
    """
    signal_energy = x.pow(2).sum().item()
    noise_energy  = (xhat - x).pow(2).sum().item()
    return 10.0 * math.log10(signal_energy / noise_energy)

@torch.no_grad()
def fft2c(x: torch.Tensor) -> torch.Tensor:  # [2,T,H,W] -> [2,T,H,W] in k-space
    xr, xi = x[0], x[1]
    xc = torch.complex(xr, xi)
    k = torch.fft.fft2(xc, norm="ortho")
    return torch.stack((k.real, k.imag), dim=0)

@torch.no_grad()
def charb_l1(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> float:
    diff = x - y
    return (diff.pow(2) + eps * eps).sqrt().mean().item()

@torch.no_grad()
def psnr_mag(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    xm, ym = complex_mag(x), complex_mag(y)
    mse = (xm - ym).pow(2).mean().item()
    if mse <= eps:
        return 99.0
    max_I = math.sqrt(2.0)  # safe bound for |complex|
    return 10.0 * math.log10((max_I * max_I) / mse)

@torch.no_grad()
def gaussian_phase_rotate(x: torch.Tensor, theta_rad: float) -> torch.Tensor:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    xr, xi = x[0], x[1]
    yr = c * xr - s * xi
    yi = s * xr + c * xi
    return torch.stack([yr, yi], dim=0)

@torch.no_grad()
def roll_xy(x: torch.Tensor, dy: int = 1, dx: int = 1) -> torch.Tensor:
    return torch.roll(x, shifts=(0, 0, dy, dx), dims=(0, 1, 2, 3))

@torch.no_grad()
def frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = img_1hw.detach().float().cpu()
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p)
    hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

def make_grid_from_channels(maps: torch.Tensor, max_ch: int = 16) -> torch.Tensor:
    C, H, W = maps.shape
    k = min(C, max_ch)
    cols = int(min(8, k))
    rows = int((k + cols - 1) // cols)
    tiles = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < k:
                m = maps[idx]
                m = (m - m.min()) / (m.max() - m.min() + 1e-6)
            else:
                m = torch.zeros((H, W), device=maps.device, dtype=maps.dtype)
            row_imgs.append(m)
        tiles.append(torch.cat(row_imgs, dim=-1))
    return torch.cat(tiles, dim=-2)  # [rows*H, cols*W]

def save_video_from_1t(frames_1t_hw: torch.Tensor, path: str, fps: int = 7):
    import imageio.v2 as iio
    T = int(frames_1t_hw.shape[1])
    frames = [frame_to_uint8(frames_1t_hw[:, t]) for t in range(T)]
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".gif"): path = path + ".mp4"
    try:
        iio.mimsave(path, frames, fps=fps)
    except Exception:
        path = os.path.splitext(path)[0] + ".gif"
        iio.mimsave(path, frames, fps=fps)
    return path

@torch.no_grad()
def save_error_visuals(x: torch.Tensor, xhat: torch.Tensor, out_dir: str, sample_id: int, is_video: bool):
    """
    Save error maps:
      - magnitude error: | |x| - |xhat| |
      - complex residual magnitude: |x - xhat|
    x, xhat: [2,T,H,W]
    """
    import imageio.v2 as iio
    xm   = complex_mag(x)      # [1,T,H,W]
    xhm  = complex_mag(xhat)   # [1,T,H,W]
    err_mag  = (xm - xhm).abs()                # [1,T,H,W]
 
    if is_video:
        mid = int(xm.shape[1] // 2)
        iio.imwrite(os.path.join(out_dir, f"sample{sample_id:03d}_errmag_mid.png"),
                    to_uint8_abs(err_mag[:, mid]))
        # full error videos
        save_video_from_1t(err_mag,  os.path.join(out_dir, f"sample{sample_id:03d}_errmag.mp4"),  fps=7)
    else:
        iio.imwrite(os.path.join(out_dir, f"sample{sample_id:03d}_errmag.png"),
                    to_uint8_abs(err_mag[:, 0]))

# ======================= Smoothness & Gaussianity =======================
@torch.no_grad()
def tv3d(z: torch.Tensor) -> float:  # z: [Cz, n, H, W]
    parts = []
    if z.size(1) > 1: parts.append((z[:, 1:] - z[:, :-1]).abs().mean())
    if z.size(2) > 1: parts.append((z[:, :, 1:] - z[:, :, :-1]).abs().mean())
    if z.size(3) > 1: parts.append((z[:, :, :, 1:] - z[:, :, :, :-1]).abs().mean())
    if not parts: return 0.0
    return (sum(parts) / len(parts)).item()

@torch.no_grad()
def temporal_jerk(z: torch.Tensor) -> float:
    if z.shape[1] < 3: return float("nan")
    j = (z[:, 2:] - 2*z[:, 1:-1] + z[:, :-2]).abs().mean()
    return j.item()

@torch.no_grad()
def channel_gaussianity_stats(mu_all: torch.Tensor) -> Dict[str, float]:
    N, C = mu_all.shape
    if N < 20:
        return {"jb_reject_rate": float("nan"), "skew_abs_mean": float("nan"),
                "kurtosis_mean": float("nan"), "std_mean": float("nan"), "mean_abs_mean": float("nan")}
    mean = mu_all.mean(dim=0); std = mu_all.std(dim=0).clamp_min(1e-6)
    Z = (mu_all - mean) / std
    skew = (Z.pow(3).mean(dim=0))
    kurt = (Z.pow(4).mean(dim=0))
    JB = (N / 6.0) * (skew.pow(2) + (kurt - 3.0).pow(2) * 0.25)
    reject = (JB > 5.991).float().mean().item()  # chi2(df=2), α=0.05
    return {
        "skew_abs_mean": skew.abs().mean().item(),
        "kurtosis_mean": kurt.mean().item(),
        "jb_reject_rate": reject,
        "std_mean": std.mean().item(),
        "mean_abs_mean": mean.abs().mean().item(),
    }

@torch.no_grad()
def offdiag_cov_abs_mean(mu_all: torch.Tensor) -> float:
    N, C = mu_all.shape
    if N < C: return float("nan")
    cov = (mu_all.t() @ mu_all) / N
    off = cov - torch.diag(torch.diag(cov))
    return off.abs().mean().item()

def save_channel_histograms(Z_all_cpu: torch.Tensor, out_dir: str, max_ch: int = 16, bins: int = 51):
    """
    Z_all_cpu: [Npos, C] standardized per-channel (zero-mean/unit-std within-sample already applied)
    Saves a grid of histograms with N(0,1) overlay to visually assess Gaussianity.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available for histograms: {e}")
        return
    Z = Z_all_cpu.numpy()
    C = Z.shape[1]
    k = min(max_ch, C)
    cols = min(8, k)
    rows = int((k + cols - 1) // cols)
    xs = np.linspace(-4.0, 4.0, 401)
    normal_pdf = 1.0/np.sqrt(2*np.pi) * np.exp(-0.5*xs*xs)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.0), squeeze=False)
    for i in range(k):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.hist(Z[:, i], bins=bins, density=True, alpha=0.85)
        ax.plot(xs, normal_pdf, linewidth=1.0)
        ax.set_title(f"ch {i}", fontsize=8)
        ax.set_xlim(-4.0, 4.0)
        ax.set_yticks([])
    # hide unused
    for j in range(k, rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "gauss_channel_histograms.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved channel Gaussianity histograms -> {path}")

# ======================= Locality (decoder PSF) =======================
@torch.no_grad()
def locality_psf_metrics(decoder_fn, z: torch.Tensor, eps: float = 0.1, probes: int = 4) -> Dict[str, float]:
    """
    decoder_fn: callable(Z)->patch output [2,t,h,w] or [1,2,t,h,w]
    z: [1,Cz,n,H',W'] (one patch latent)
    Returns temporal std (frames) and spatial RMS radius of |Δx|.
    """
    device = z.device
    out0 = decoder_fn(z)
    if isinstance(out0, list): 
        out0 = out0[0]
    if out0.dim() == 5: 
        out0 = out0.squeeze(0)  # -> [2,t,h,w]

    base_mag = complex_mag(out0).squeeze(0)  # [t,h,w]
    T, H, W = base_mag.shape

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    temp_spreads, spatial_rms = [], []
    Cz, n, Hh, Ww = z.shape[1:]

    for _ in range(probes):
        c = random.randrange(Cz); tt = random.randrange(n); h = random.randrange(Hh); w = random.randrange(Ww)
        z2 = z.clone()
        z2[0, c, tt, h, w] = z2[0, c, tt, h, w] + eps
        out = decoder_fn(z2)
        if isinstance(out, list): 
            out = out[0]
        if out.dim() == 5: 
            out = out.squeeze(0)

        diff = (complex_mag(out).squeeze(0) - base_mag).abs()  # [t,h,w]

        e_t = diff.sum(dim=(1,2)) + 1e-12
        w_t = e_t / e_t.sum()
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        mu_t = (w_t * t_idx).sum()
        var_t = (w_t * (t_idx - mu_t) ** 2).sum()
        temp_spreads.append(torch.sqrt(var_t).item())

        t_star = int(torch.argmax(e_t).item())

        E = diff[t_star]**2          # energy
        E_sum = E.sum() + 1e-12
        e_hw = (E / E_sum)

        # Optional robust trimming (keep e.g. 99% of the energy):
        keep = 0.99
        flat = torch.sort(E.flatten(), descending=True).values
        th = flat.cumsum(0)
        th = flat[(th <= keep * E_sum).sum().clamp(min=1)-1]  # threshold at 99% cum energy
        mask = (E >= th)
        e_hw = (E * mask) / (E * mask).sum().clamp_min(1e-12)
        
        cx = (e_hw * xx).sum(); cy = (e_hw * yy).sum()
        r2 = (xx - cx)**2 + (yy - cy)**2
        spatial_rms.append(torch.sqrt((e_hw * r2).sum()).item())

    return {
        "locality_temporal_spread_frames_mean": float(np.mean(temp_spreads)) if temp_spreads else float("nan"),
        "locality_spatial_rms_px_mean": float(np.mean(spatial_rms)) if spatial_rms else float("nan"),
    }

# ======================= Robustness (full patch pipeline) =======================
@torch.no_grad()
def robustness_metrics_full_pipeline(model, x_full: torch.Tensor, patch_size, patch_bs: int,
                                     noise_sigma: float = 0.02, phase_deg: float = 5.0) -> Dict[str, float]:
    """
    Measures robustness through the SAME patchify->model->depatchify pipeline.
    Latent change is based on concatenating all patch μ vectors (deterministic μ-probe).
    """
    device = x_full.device
    _, T, H, W = x_full.shape
    P_t, P_h, P_w = patch_size

    def _forward_full(x):
        r_p, strides = patchify(x[0], patch_size=(P_t if T>1 else 1, P_h, P_w))
        i_p, _       = patchify(x[1], patch_size=(P_t if T>1 else 1, P_h, P_w))
        Np = int(r_p.shape[0])
        patches = [ torch.stack((r_p[n], i_p[n]), dim=0).to(device) for n in range(Np) ]
        xhat_list, mu_list = [], []
        for i in range(0, Np, patch_bs):
            chunk = patches[i:i+patch_bs]
            out = model(chunk)  # -> (xhats, mus, logvs, zs)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                xhats, mus = out[0], out[1]
            else:
                raise RuntimeError("Model forward must return (xhats, mus, logvs, zs)")
            for xh, mu in zip(xhats, mus):
                xhat_list.append(xh.squeeze(0))
                mu_list.append(mu.squeeze(0))
        r_rec = torch.stack([xh[0] for xh in xhat_list], dim=0)
        i_rec = torch.stack([xh[1] for xh in xhat_list], dim=0)
        xhat_r = depatchify(r_rec, (T, H, W), (P_t if T>1 else 1, P_h, P_w), strides)
        xhat_i = depatchify(i_rec, (T, H, W), (P_t if T>1 else 1, P_h, P_w), strides)
        xhat   = torch.stack((xhat_r, xhat_i), dim=0)
        return xhat, mu_list

    def _encode_vector(x):
        _, mu_list = _forward_full(x)
        flats = [mu.reshape(-1) for mu in mu_list]
        return torch.cat(flats, dim=0)

    _ = _forward_full(x_full)  # warmup path/sizes
    zvec0 = _encode_vector(x_full)

    xn = (x_full + noise_sigma * torch.randn_like(x_full)).clamp_(-1, 1)
    xp = gaussian_phase_rotate(x_full, math.radians(phase_deg))
    xr = roll_xy(x_full, 1, 1)

    def _rel_latent_change(xp):
        zv = _encode_vector(xp)
        num = (zv - zvec0).pow(2).sum().sqrt()
        den = zvec0.pow(2).sum().sqrt().clamp_min(1e-6)
        return (num / den).item()

    def _psnr(xp):
        xh, _ = _forward_full(xp)
        return snr_complex(x_full, xh)

    return {
        "robust_latent_relchange_noise": _rel_latent_change(xn),
        "robust_latent_relchange_phase": _rel_latent_change(xp),
        "robust_latent_relchange_roll":  _rel_latent_change(xr),
        "robust_snr_noise": _psnr(xn),
        "robust_snr_phase": _psnr(xp),
        "robust_snr_roll":  _psnr(xr),
    }


# ======================= Optional LPIPS =======================
_HAS_LPIPS = False
try:
    import lpips
    _HAS_LPIPS = True
    _LPIPS_NET = lpips.LPIPS(net="alex").eval()
    for p in _LPIPS_NET.parameters():
        p.requires_grad_(False)
except Exception:
    _HAS_LPIPS = False
    _LPIPS_NET = None

@torch.no_grad()
def _to_lpips_img(mag_1hw: torch.Tensor) -> torch.Tensor:
    # Map magnitude to [-1,1] with a fixed cap (<= sqrt(2))
    cap = math.sqrt(2.0)
    x = torch.nan_to_num(mag_1hw, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x / cap).clamp_(0, 1) * 2.0 - 1.0
    return x.repeat(3, 1, 1).unsqueeze(0)

@torch.no_grad()
def lpips_on_full(x: torch.Tensor, xhat: torch.Tensor) -> float:
    if not _HAS_LPIPS: 
        return float("nan")
    xm  = complex_mag(x)
    xhm = complex_mag(xhat)
    T = xm.shape[1]
    t = T // 2  # middle frame
    A = _to_lpips_img(xm[:, t])
    B = _to_lpips_img(xhm[:, t])
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return float(_LPIPS_NET(A, B).mean().item())

@torch.no_grad()
def to_uint8_abs(img_1hw: torch.Tensor, cap: float = 0.1) -> np.ndarray:
    """
    Absolute (fixed-range) rendering for error maps.
    img_1hw: [1,H,W] nonnegative (we'll clamp)
    cap: values >= cap map to white; no percentile stretching.
    """
    f = img_1hw.detach().float().cpu()
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = f.clamp_min_(0.0)
    cap = max(float(cap), 1e-8)
    g = (f / cap).clamp_(0, 1)
    g = (g * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

# ======================= Main evaluation =======================
def dynamic_import(import_path: str, class_name: str):
    mod = __import__(import_path, fromlist=[class_name])
    return getattr(mod, class_name)

def detect_mode(cfg: Dict[str, Any], sample_tensor: torch.Tensor = None) -> str:
    try:
        m = cfg.get("stages", {}).get("mode", None)
        if m in ("pretrain_2d", "videos"): return m
    except Exception:
        pass
    try:
        m = cfg.get("test_dataset", {}).get("args", {}).get("stage_mode", None)
        if m in ("pretrain_2d", "videos"): return m
    except Exception:
        pass
    if sample_tensor is not None and sample_tensor.ndim == 4:
        return "pretrain_2d" if sample_tensor.shape[1] == 1 else "videos"
    return "videos"

@torch.no_grad()
def evaluate(cfg: Dict[str, Any], ckpt_path: str, out_dir: str,
             max_items: int = 64, probes: int = 4, locality_eps: float = 0.1,
             noise_sigma: float = 0.02, phase_deg: float = 5.0,
             vis_max: int = 8, hist_max_ch: int = 16, hist_bins: int = 51):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if _HAS_LPIPS and _LPIPS_NET is not None:
        _LPIPS_NET.to(device)

    # ---- Dataset (TEST) ----
    dl_cfg = cfg["dataloader"]
    test_ds = CINEDataset(**cfg["test_dataset"]["args"])
    test_dl = DataLoader(
        test_ds,
        batch_size=dl_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        collate_fn=ragged_collate,
        drop_last=False,
    )

    # ---- Model ----
    M = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = M(**cfg["model"]["args"]).to(device).eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[eval] loaded: {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[eval] WARNING: checkpoint not found: {ckpt_path}")

    # ---- Patch params (mirror validation defaults) ----
    vcfg = cfg.get("validation", {})
    patch_h = int(vcfg.get("patch_h", 80))
    patch_w = int(vcfg.get("patch_w", 80))
    patch_t_default = int(vcfg.get("patch_t", 11))  # used if T>1
    patch_bs = int(vcfg.get("patch_batch", 32))

    # ---- Accumulators ----
    per_sample_rows = []
    acc_lists = {k: [] for k in [
        "l1_complex",
        "l1k_complex",
        "snr_complex",
        "lpips_middle",
        "smooth_tv3d",
        "smooth_temporal_jerk",
        "locality_temporal_spread_frames_mean",
        "locality_spatial_rms_px_mean",
        "robust_latent_relchange_noise",
        "robust_latent_relchange_phase",
        "robust_latent_relchange_roll",
        "robust_snr_noise",
        "robust_snr_phase",
        "robust_snr_roll",
    ]}
    MU_chunks = []  # collect raw mu (not standardized) across patches/samples

    # ---- Iterate ----
    n_done = 0
    mode_known = None
    for batch in tqdm(test_dl, desc="eval"):
        for x in batch:
            x = x.to(device=device, dtype=torch.float32)  # [2,T,H,W]
            _, T, H, W = x.shape
            if mode_known is None:
                mode_known = detect_mode(cfg, x)

            is_video = (mode_known == "videos") and (T > 1)
            patch_t = (patch_t_default if is_video else 1)
            patch_size = (patch_t, patch_h, patch_w)

            # --- Patchify ---
            r_p, strides = patchify(x[0], patch_size)
            i_p, _       = patchify(x[1], patch_size)
            Np = int(r_p.shape[0])
            patches = [ torch.stack((r_p[n], i_p[n]), dim=0).to(device) for n in range(Np) ]

            # --- Forward (micro-batches) ---
            xhat_list, mu_list, logv_list, z_list = [], [], [], []
            for i in range(0, Np, patch_bs):
                chunk = patches[i:i+patch_bs]
                out = model(chunk)  # -> (xhats, mus, logvs, zs)
                if not (isinstance(out, (list, tuple)) and len(out) >= 4):
                    raise RuntimeError("Model forward must return (xhats, mus, logvs, zs)")
                xhats, mus, logvs, zs = out
                for xh, mu, lv, z in zip(xhats, mus, logvs, zs):
                    xhat_list.append(xh.squeeze(0))  # [2,t,h,w]
                    mu_list.append(mu.squeeze(0))    # [Cz,n,H',W']
                    logv_list.append(lv.squeeze(0))
                    z_list.append(z.squeeze(0))

            # --- Depatchify to full recon ---
            r_rec = torch.stack([xh[0] for xh in xhat_list], dim=0)
            i_rec = torch.stack([xh[1] for xh in xhat_list], dim=0)
            xhat_r = depatchify(r_rec, (T, H, W), patch_size, strides)
            xhat_i = depatchify(i_rec, (T, H, W), patch_size, strides)
            xhat   = torch.stack((xhat_r, xhat_i), dim=0)  # [2,T,H,W]

            # --- Full-image metrics ---
            snr_c  = snr_complex(x, xhat)
            l1_img = charb_l1(x, xhat)
            l1k    = charb_l1(fft2c(x), fft2c(xhat))
            lpips_m = lpips_on_full(x, xhat)

            # --- Smoothness (μ over ALL patches concatenated along time) ---
            mu_cat = torch.cat(mu_list, dim=1) if len(mu_list) > 0 else None  # [Cz, sum_n, H', W']
            tv = tv3d(mu_cat) if mu_cat is not None else float("nan")
            jerk = temporal_jerk(mu_cat) if mu_cat is not None else float("nan")

            # --- Locality (decoder PSF) using ONE representative patch ---
            if hasattr(model, "decode") and len(mu_list) > 0:
                def decode_patch(Z):
                    out = model.decode([Z])
                    return out[0] if isinstance(out, list) else out
                z_one = mu_list[0].unsqueeze(0)  # [1,Cz,n,H',W']
                loc = locality_psf_metrics(decode_patch, z_one, eps=locality_eps, probes=probes)
            else:
                loc = {"locality_temporal_spread_frames_mean": float("nan"),
                       "locality_spatial_rms_px_mean": float("nan")}

            # --- Robustness through FULL patch pipeline ---
            rob = robustness_metrics_full_pipeline(
                model, x, patch_size=patch_size, patch_bs=patch_bs,
                noise_sigma=noise_sigma, phase_deg=phase_deg
            )

            # --- Gaussianity accumulation (store RAW mu, standardize later globally) ---
            subsample_positions = int(cfg.get("gauss_subsample_positions", 8192))
            for mu in mu_list:
                C, nH, Hh, Ww = mu.shape
                M = nH * Hh * Ww
                mu_flat = mu.reshape(C, M).t().contiguous()  # [M, C] raw μ (no standardization)
                if mu_flat.shape[0] > subsample_positions:
                    idx = torch.randperm(mu_flat.shape[0], device=mu_flat.device)[:subsample_positions]
                    mu_flat = mu_flat[idx]
                MU_chunks.append(mu_flat.detach().cpu())

            # --- Visualizations (capped by vis_max) ---
            save_vis = (n_done < int(vis_max))
            if save_vis:
                import imageio.v2 as iio
                xm  = complex_mag(x)    # [1,T,H,W]
                xhm = complex_mag(xhat) # [1,T,H,W]

                if is_video:
                    mid = T // 2
                    iio.imwrite(os.path.join(out_dir, f"sample{n_done:03d}_gt_mid.png"),
                                frame_to_uint8(xm[:, mid]))
                    iio.imwrite(os.path.join(out_dir, f"sample{n_done:03d}_recon_mid.png"),
                                frame_to_uint8(xhm[:, mid]))
                    save_video_from_1t(xm,  os.path.join(out_dir, f"sample{n_done:03d}_gt_mag.mp4"),  fps=7)
                    save_video_from_1t(xhm, os.path.join(out_dir, f"sample{n_done:03d}_recon_mag.mp4"), fps=7)
                else:
                    err = (xm[:, 0] - xhm[:, 0]).abs()
                    iio.imwrite(os.path.join(out_dir, f"sample{n_done:03d}_gt.png"),    frame_to_uint8(xm[:, 0]))
                    iio.imwrite(os.path.join(out_dir, f"sample{n_done:03d}_recon.png"), frame_to_uint8(xhm[:, 0]))

                # --- error map visualizations ---
                try:
                    save_error_visuals(x, xhat, out_dir, n_done, is_video=is_video)
                except Exception as e:
                    print(f"[viz] error-map save failed on sample {n_done}: {e}")

                # latent μ grid (image or latent-video over latent frames)
                if len(mu_list) > 0:
                    mu0 = mu_list[0]  # [Cz,n,H',W']
                    if is_video and mu0.shape[1] > 1:
                        # latent video grid across n (latent frames) for a representative patch
                        frames = []
                        for t in range(mu0.shape[1]):
                            grid = make_grid_from_channels(mu0[:, t]).detach().cpu().numpy()
                            frames.append((grid * 255.0).astype(np.uint8))
                        path = os.path.join(out_dir, f"sample{n_done:03d}_latent_mu_grid.mp4")
                        try:
                            iio.mimsave(path, frames, fps=6)
                        except Exception:
                            iio.mimsave(path.replace(".mp4", ".gif"), frames, fps=6)
                    else:
                        t_m = 0
                        grid = make_grid_from_channels(mu0[:, t_m]).detach().cpu().numpy()
                        iio.imwrite(os.path.join(out_dir, f"sample{n_done:03d}_latent_mu_grid_t{t_m}.png"),
                                    (grid * 255.0).astype(np.uint8))

                # PSF panel (optional)
                if hasattr(model, "decode") and len(mu_list) > 0:
                    try:
                        mu0 = mu_list[0]; z_one = mu0.unsqueeze(0)
                        def decode_patch(Z):
                            out = model.decode([Z]); 
                            return out[0] if isinstance(out, list) else out
                        base = decode_patch(z_one); 
                        if base.dim() == 5: base = base.squeeze(0)
                        base_mag = complex_mag(base).squeeze(0)  # [t,h,w]
                        z2 = z_one.clone()
                        z2[0, 0, min(0, z2.shape[2]-1), z2.shape[3]//2, z2.shape[4]//2] += locality_eps
                        out = decode_patch(z2); 
                        if out.dim() == 5: out = out.squeeze(0)
                        diff = (complex_mag(out).squeeze(0) - base_mag).abs()
                        t_star = int(torch.argmax(diff.sum(dim=(1,2))).item())
                        im = diff[t_star].detach().cpu().numpy()
                        im = im / (im.max() + 1e-6)
                        from imageio.v2 import imwrite
                        imwrite(os.path.join(out_dir, f"sample{n_done:03d}_psf_t{t_star}.png"),
                                (im * 255.0).astype(np.uint8))
                    except Exception as e:
                        print(f"[viz] PSF viz skipped on sample {n_done}: {e}")

            # --- Record row ---
            row = {
                "mode": mode_known,
                "T": int(T), "H": int(H), "W": int(W),
                "snr_complex": snr_c,
                "l1_complex": l1_img,
                "l1k_complex": l1k,
                "lpips_middle": lpips_m,
                "smooth_tv3d": tv,
                "smooth_temporal_jerk": jerk,
                **loc, **rob,
            }

            per_sample_rows.append(row)
            for k in acc_lists.keys():
                if k in row and row[k] is not None:
                    acc_lists[k].append(row[k])

            n_done += 1
            if n_done >= max_items:
                break
        if n_done >= max_items:
            break

    # ---- Gaussianity across channels (aggregate; two-pass GLOBAL standardization) ----
    if len(MU_chunks) > 0:
        MU_all = torch.cat(MU_chunks, dim=0)  # [Npos, C], raw μ
        mean = MU_all.mean(dim=0, keepdim=True)
        std  = MU_all.std(dim=0, keepdim=True).clamp_min(1e-6)
        Z_all = (MU_all - mean) / std                        # global per-channel Z
        gauss = channel_gaussianity_stats(Z_all)
        gauss["cov_offdiag_abs_mean"] = offdiag_cov_abs_mean(Z_all)

        # --- Save channel histograms (uses globally standardized Z_all) ---
        try:
            save_channel_histograms(Z_all.cpu(), out_dir, max_ch=int(hist_max_ch), bins=int(hist_bins))
        except Exception as e:
            print(f"[viz] histogram save failed: {e}")
    else:
        gauss = {"skew_abs_mean": float("nan"), "kurtosis_mean": float("nan"),
                "jb_reject_rate": float("nan"), "std_mean": float("nan"),
                "mean_abs_mean": float("nan"), "cov_offdiag_abs_mean": float("nan")}

    # ---- Summaries ----
    def _nm(a): return float(np.nanmean(a)) if len(a) else float("nan")
    def _nd(a): return float(np.nanmedian(a)) if len(a) else float("nan")

    summary = { f"mean/{k}": _nm(v) for k, v in acc_lists.items() }
    summary.update({ f"median/{k}": _nd(v) for k, v in acc_lists.items() })
    summary.update({ f"gauss/{k}": v for k, v in gauss.items() })
    summary["num_items"] = n_done
    summary["mode"] = mode_known or "unknown"
    summary["lpips_used"] = bool(_HAS_LPIPS)

    # ---- Save ----
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import pandas as pd
        pd.DataFrame(per_sample_rows).to_csv(os.path.join(out_dir, "per_sample_metrics.csv"), index=False)
    except Exception:
        with open(os.path.join(out_dir, "per_sample_metrics.txt"), "w") as f:
            for r in per_sample_rows:
                f.write(json.dumps(r) + "\n")

    print("\n=== VAE evaluation summary ===")
    for k in sorted(summary.keys()):
        print(f"{k:40s} : {summary[k]}")

    return summary

# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/vae.yaml", help="YAML config path")
    ap.add_argument("--ckpt",   type=str, default="/storage/matt_models/cardiac_vae/step_0052000/state.pt", help="Path to checkpoint .pt (state.pt)")
    ap.add_argument("--out",    type=str, default="eval_out", help="Output directory")
    ap.add_argument("--max-items", type=int, default=500, help="Max number of samples to evaluate")
    ap.add_argument("--locality-probes", type=int, default=4)
    ap.add_argument("--locality-eps", type=float, default=0.1)
    ap.add_argument("--robust-noise-sigma", type=float, default=0.02)
    ap.add_argument("--robust-phase-deg", type=float, default=5.0)
    # NEW: visualization and histogram controls
    ap.add_argument("--vis-max", type=int, default=5, help="Max number of samples to visualize")
    ap.add_argument("--hist-max-ch", type=int, default=16, help="Num channels to show in Gaussianity hist figure")
    ap.add_argument("--hist-bins", type=int, default=51, help="Bins for Gaussianity histograms")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg,
             ckpt_path=args.ckpt,
             out_dir=args.out,
             max_items=args.max_items,
             probes=args.locality_probes,
             locality_eps=args.locality_eps,
             noise_sigma=args.robust_noise_sigma,
             phase_deg=args.robust_phase_deg,
             vis_max=args.vis_max,
             hist_max_ch=args.hist_max_ch,
             hist_bins=args.hist_bins)

if __name__ == "__main__":
    main()
