#!/usr/bin/env python3
import os, sys
import time
import importlib
from typing import Dict, Any, List, Tuple
import numpy as np
import math
# Add the launch (cwd) directory to Python's import path
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

import yaml
import wandb

# Dataset: keep this import path; the class should return:
# - pretrain_2d: [2, 1, H, W]
# - videos:     [2, L, H, W] with odd L
from data.cine_dataset import CINEDataset
from tqdm.auto import tqdm
import time

# Optional LPIPS (pixel space)
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# -------------------- ragged collate --------------------

def ragged_collate(batch):
    # Keep batch as a list of tensors; shapes can vary across items
    return batch


# -------------------- loss-weight schedules (per-step) --------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _linear_scale(p: float, start: float, end: float) -> float:
    p = _clamp01(p)
    return start + (end - start) * p

def _cosine_scale(p: float, start: float, end: float) -> float:
    import math
    p = _clamp01(p)
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * p))

class LossSchedulerSteps:
    """
    Per-loss weight scaler over total optimizer steps (1..total_steps).
    """
    def __init__(self, base: Dict[str, float], sched_cfg: Dict[str, Any], total_steps: int):
        self.base = base
        self.sched_cfg = sched_cfg or {}
        self.total_steps = max(1, int(total_steps))

    def _scale_for(self, name: str, step_idx: int) -> float:
        cfg = self.sched_cfg.get(name, None)
        if cfg is None:
            return 1.0
        typ = str(cfg.get("type", "linear")).lower()
        start = float(cfg.get("start_scale", 0.0))
        end   = float(cfg.get("end_scale", 1.0))
        pct   = _clamp01(float(cfg.get("pct_of_steps", 1.0)))
        cutoff = max(1, int(round(self.total_steps * pct)))
        p = min(1.0, step_idx / cutoff)
        s = _cosine_scale(p, start, end) if typ == "cosine" else _linear_scale(p, start, end)
        if step_idx >= cutoff:
            s = end
        return s

    def weights_for_step(self, step_idx: int) -> Dict[str, float]:
        return {k: self.base[k] * self._scale_for(k, step_idx) for k in self.base.keys()}


# -------------------- colleague patch/depatch helpers (verbatim) --------------------

from math import ceil

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

def depatchify(
    patches: torch.Tensor,   # (N, P0, P1, P2)
    data_shape: tuple,       # (D0, D1, D2)
    patch_size: tuple,       # (P0, P1, P2)
    strides: tuple,          # (S0, S1, S2)
):
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
        raise ValueError(
            f"patches.shape[0]={patches.shape[0]} does not match expected {expected_N} "
            f"(= {n0}*{n1}*{n2}). Check your patch ordering or counts."
        )

    O0 = max(0, P0 - S0)
    O1 = max(0, P1 - S1)
    O2 = max(0, P2 - S2)

    def axis_weights(L_eff: int, idx: int, n: int, O: int) -> torch.Tensor:
        has_prev = (idx > 0)
        has_next = (idx < n - 1)

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
            if L_left == 1:
                w[:1] = 0.5
            else:
                w[:L_left] = torch.linspace(0.0, 1.0, steps=L_left, device=device)
        if L_right > 0:
            if L_right == 1:
                w[-1:] = 0.5
            else:
                w[-L_right:] = torch.linspace(1.0, 0.0, steps=L_right, device=device)
        return w

    patch_idx = 0
    for i in range(n0):
        start0 = i * S0; end0 = min(start0 + P0, D0)
        s0  = slice(start0, end0)
        ps0 = slice(0,      end0 - start0)
        w0  = axis_weights(end0 - start0, i, n0, O0)

        for j in range(n1):
            start1 = j * S1; end1 = min(start1 + P1, D1)
            s1  = slice(start1, end1)
            ps1 = slice(0,      end1 - start1)
            w1  = axis_weights(end1 - start1, j, n1, O1)

            for k in range(n2):
                start2 = k * S2; end2 = min(start2 + P2, D2)
                s2  = slice(start2, end2)
                ps2 = slice(0,      end2 - start2)
                w2  = axis_weights(end2 - start2, k, n2, O2)

                w = (w0[:, None, None] * w1[None, :, None] * w2[None, None, :])

                patch = patches[patch_idx][ps0, ps1, ps2]

                out_num[s0, s1, s2] += patch * w.to(dtype)
                out_den[s0, s1, s2] += w
                patch_idx += 1

    zero_mask = (out_den == 0)
    if zero_mask.any():
        out_den[zero_mask] = 1.0
    return out_num / out_den.to(dtype)

def compute_N(data_shape, patch_size):
    _, N, _ = compute_strides_and_N(data_shape, patch_size)
    return N


# -------------------- Trainer (stage-aware, step-based, ragged) --------------------

class CardiacVAETrainer:
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        cfg['logging']['out_dir'] = cfg['logging']['out_dir'] + f"/{cfg['stages']['mode']}"
        self.cfg = cfg
        self.model = model

        self.model = torch.compile(self.model) # Can do w/ fixed shapes..

        # Stage
        mode = cfg["stages"]["mode"]
        assert mode in ("pretrain_2d", "videos")
        self.mode = mode

        proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        self.accelerator = Accelerator(project_config=proj_cfg, kwargs_handlers=[ddp_kwargs])  # float32

        opt_cfg = cfg["optim"]
        self.total_steps = int(opt_cfg["total_steps"])
        self.accum_steps = int(opt_cfg.get("accum_steps", 1))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]), weight_decay=opt_cfg["weight_decay"]
        )

        self.scheduler = None
        if opt_cfg.get("scheduler", {}).get("type", "none") == "cosine":
            eta_min = opt_cfg["lr"] * opt_cfg["scheduler"].get("eta_min_ratio", 0.1)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_steps * 4, eta_min=eta_min
            )

        self.train_dl = train_dl
        self.val_dl   = val_dl

        self.global_step = 0

        stage_loss_cfg = cfg["loss"][self.mode]
        if self.mode == "pretrain_2d":
            base_weights = {
                "img": float(stage_loss_cfg.get("w_img", 3.0)),
                "lpips": float(stage_loss_cfg.get("w_lpips", 3.0)),
                "kl": float(stage_loss_cfg.get("w_kl", 3e-6)),
            }
        else:
            base_weights = {
                "img": float(stage_loss_cfg.get("w_img", 1.0)),
                "lpips": float(stage_loss_cfg.get("w_lpips", 0.5)),
                "kl": float(stage_loss_cfg.get("w_kl", 1e-3)),
                "tvz": float(stage_loss_cfg.get("w_tvz", 1e-3)),
            }
        self.loss_sched = LossSchedulerSteps(base_weights, stage_loss_cfg.get("schedules", {}), self.total_steps)
        self.current_w = self.loss_sched.weights_for_step(1)

        if self.cfg["model"].get("load_state_dict_from", None) is not None and self.cfg['model']['resume']:
            resume_state = torch.load(self.cfg["model"]["load_state_dict_from"], map_location="cpu")
            self.optimizer.load_state_dict(resume_state["optimizer"])
            if self.scheduler is not None and "scheduler" in resume_state:
                self.scheduler.load_state_dict(resume_state["scheduler"])
            self.global_step = int(resume_state.get("global_step", 0))
            self.current_w = self.loss_sched.weights_for_step(self.global_step)
            self.accelerator.print(
                f"[resume] loaded model/optim{'/sched' if self.scheduler else ''} "
                f"from step {self.global_step}"
            )

        self.start_step = self.global_step

        if self.scheduler is not None:
            (self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler)
        else:
            (self.model, self.optimizer, self.train_dl, self.val_dl) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl)

        self.grad_clip = float(opt_cfg.get("grad_clip", 0.0))

        wdir  = self.cfg["logging"].get("wandb_dir", None)
        wcache = self.cfg["logging"].get("wandb_cache_dir", None)

        if wdir:
            os.environ["WANDB_DIR"] = str(wdir)
        if wcache:
            os.environ["WANDB_CACHE_DIR"] = str(wcache)

        if self.accelerator.is_main_process:
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"].get("run_name"),
                config=cfg,
                dir=wdir,
            )

        self.use_lpips = bool(stage_loss_cfg.get("use_lpips", True))
        self.lpips_frames = str(stage_loss_cfg.get("lpips_frames", "middle")) if self.mode == "videos" else "middle"
        if self.use_lpips and not _HAS_LPIPS:
            self.accelerator.print("[WARN] LPIPS requested but 'lpips' package not found; disabling LPIPS.")
            self.use_lpips = False
        self.lpips_net = None
        if self.use_lpips:
            self.lpips_net = lpips.LPIPS(net="alex")
            self.lpips_net = self.lpips_net.to(self.accelerator.device)
            self.lpips_net.eval()
            for p in self.lpips_net.parameters():
                p.requires_grad_(False)

        self.model.to(torch.float32)

        # --------- validation patch config ----------
        vcfg = self.cfg.get("validation", {})
        self.val_patch_h = int(vcfg.get("patch_h", 80))
        self.val_patch_w = int(vcfg.get("patch_w", 80))
        self.val_patch_t = int(vcfg.get("patch_t", 11))      # videos
        self.val_patch_bs = int(vcfg.get("patch_batch", 32)) # how many patches per forward

    # ---------- helpers ----------
    @staticmethod
    def _fft2c(x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[0] == 2, f"fft2c expects [2,T,H,W], got {tuple(x.shape)}"
        xr, xi = x[0], x[1]
        xc = torch.complex(xr, xi)
        k = torch.fft.fft2(xc, norm="ortho")
        return torch.stack((k.real, k.imag), dim=0)

    @staticmethod
    def _complex_mag(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[0] == 2, f"complex_mag expects [2,T,H,W], got {tuple(x.shape)}"
        return (x.pow(2).sum(dim=0, keepdim=True) + eps).sqrt()

    @staticmethod
    def _charb_l1_mag(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        diff = CardiacVAETrainer._complex_mag(x) - CardiacVAETrainer._complex_mag(y)
        return (diff.pow(2) + eps * eps).sqrt().mean()

    @staticmethod
    def _charb_l1(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        diff = x - y
        return (diff.pow(2) + eps * eps).sqrt().mean()

    @staticmethod
    def _tv3d(z: torch.Tensor) -> torch.Tensor:
        parts = []
        if z.size(1) > 1:
            parts.append((z[:, 1:] - z[:, :-1]).abs().mean())
        if z.size(2) > 1:
            parts.append((z[:, :, 1:] - z[:, :, :-1]).abs().mean())
        if z.size(3) > 1:
            parts.append((z[:, :, :, 1:] - z[:, :, :, :-1]).abs().mean())
        if not parts:
            return z.new_zeros(())
        return sum(parts) / len(parts)

    @staticmethod
    def _kl_gaussian_channels(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        logvar = log_var.clamp(-30.0, 20.0)
        return 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0).mean()

    @staticmethod
    def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
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

    def _to_wandb_video_one(self, x_mag_1t: torch.Tensor, fps: int = 7):
        T = int(x_mag_1t.shape[1])
        frames = [ self._frame_to_uint8(x_mag_1t[:, t]) for t in range(T) ]
        arr = np.stack(frames, axis=0)
        arr = arr[:, None, :, :]
        arr = np.repeat(arr, 3, axis=1)
        return wandb.Video(arr, fps=fps, format="mp4")

    # ------ LPIPS helpers ------
    @staticmethod
    def _to_lpips_img(mag_1hw: torch.Tensor) -> torch.Tensor:
        # Fixed-range mapping: 0 -> -1,  (max_mag) -> +1
        # Use your analysis suggestion; clamp to sqrt(2) for safety.
        max_mag = 1.0885  # from your JSON: recommendations.lpips_max_mag_suggestion
        max_mag = min(max_mag, math.sqrt(2.0))

        x = torch.nan_to_num(mag_1hw, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x / max_mag).clamp_(0, 1) * 2.0 - 1.0
        return x.repeat(3, 1, 1).unsqueeze(0)

    def _lpips_loss_item(self, x: torch.Tensor, xhat: torch.Tensor, mode: str) -> torch.Tensor:
        if not self.use_lpips:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        def _call_lpips(imgA_1hw, imgB_1hw):
            A = self._to_lpips_img(imgA_1hw).detach()
            B = self._to_lpips_img(imgB_1hw)
            A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
            return self.lpips_net(A, B).mean()

        if mode == "pretrain_2d":
            xm  = self._complex_mag(x)[:, 0]
            xhm = self._complex_mag(xhat)[:, 0]
            return _call_lpips(xm, xhm)

        xm  = self._complex_mag(x)
        xhm = self._complex_mag(xhat)
        T = xm.shape[1]
        choice = self.lpips_frames

        if choice == "all" or choice.startswith("all_stride"):
            stride = 1
            if choice.startswith("all_stride"):
                try:
                    stride = max(1, int(choice.split("_")[-1]))
                except Exception:
                    stride = 1
            idxs = torch.arange(0, T, step=stride, device=x.device)
            CHUNK = 32
            vals = []
            for start in range(0, idxs.numel(), CHUNK):
                end = min(start + CHUNK, idxs.numel())
                batchA, batchB = [], []
                for t in idxs[start:end]:
                    A = self._to_lpips_img(xm[:, int(t)]).detach()
                    B = self._to_lpips_img(xhm[:, int(t)])
                    batchA.append(A); batchB.append(B)
                A = torch.cat(batchA, dim=0); B = torch.cat(batchB, dim=0)
                A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
                B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
                vals.append(self.lpips_net(A, B).mean())
            return torch.stack(vals).mean()

        if choice == "middle":
            t = T // 2
            return _call_lpips(xm[:, t], xhm[:, t])
        elif choice == "sample2":
            if T >= 2:
                idxs = torch.randint(low=0, high=T, size=(2,), device=x.device)
            else:
                idxs = torch.tensor([0, 0], device=x.device)
            return torch.stack([_call_lpips(xm[:, int(t)], xhm[:, int(t)]) for t in idxs]).mean()
        else:
            t = 0 if T == 1 else torch.randint(low=0, high=T, size=(1,), device=x.device).item()
            return _call_lpips(xm[:, t], xhm[:, t])

    # ---------- diagnostics (validation) ----------
    def _radial_band_errors_one(self, x: torch.Tensor, xhat: torch.Tensor,
                                bands=((0.0, 1/3), (1/3, 2/3), (2/3, 1.0))) -> dict:
        device = x.device
        _, T, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        r = (xx**2 + yy**2).sqrt()
        r = r / r.max().clamp_min(1e-6)

        kx = self._fft2c(x)
        kh = self._fft2c(xhat)
        mag_err = (self._complex_mag(kx) - self._complex_mag(kh)).abs().squeeze(0)

        out = {}
        for i, (a, b) in enumerate(bands):
            mask = (r >= a) & (r < b)
            val = (mag_err[:, mask]).mean() if mask.any() else torch.tensor(0.0, device=device)
            out[f"radial_err_band{i}"] = val
        return out

    def _latent_stats(self, mu: torch.Tensor, logv: torch.Tensor, active_thr: float = 0.02) -> dict:
        C, n, H, W = mu.shape
        X = mu.reshape(C, -1)
        mean = X.mean(dim=1)
        std = X.std(dim=1).clamp_min(1e-6)
        Z = (X - mean[:, None]) / std[:, None]
        skew = (Z**3).mean(dim=1)
        kurt = (Z**4).mean(dim=1)
        cov = (Z @ Z.t()) / Z.shape[1]
        offdiag = cov - torch.diag(torch.diag(cov))
        offdiag_mean_abs = offdiag.abs().mean()
        kl_per_voxel = 0.5 * (mu.pow(2) + logv.exp() - logv - 1.0)
        kl_per_ch = kl_per_voxel.mean(dim=(1,2,3))
        active = (kl_per_ch > active_thr).float().mean()
        return {
            "mu_mean_abs": mean.abs().mean(),
            "mu_std_mean": std.mean(),
            "mu_skew_abs_mean": skew.abs().mean(),
            "mu_kurtosis_mean": kurt.mean(),
            "cov_offdiag_mean_abs": offdiag_mean_abs,
            "active_frac": active,
        }

    @staticmethod
    def _stack_horizontal(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat([f for f in frames], dim=-1)

    @staticmethod
    def _make_grid_from_channels(maps: torch.Tensor, max_ch: int = 16) -> torch.Tensor:
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
        return torch.cat(tiles, dim=-2)

    def _log_latent_images(self, mu: torch.Tensor, z: torch.Tensor, step: int, prefix: str = "vis/latent"):
        C, n, H, W = mu.shape
        t_mid = 0 if n == 1 else (n // 2)
        mu_grid = self._make_grid_from_channels(mu[:, t_mid])
        mu_img = mu_grid.detach().cpu().numpy()
        wandb.log({f"{prefix}/mu_grid": wandb.Image(mu_img, caption=f"mu @ t={t_mid}")}, step=step)
        for ch in range(min(3, C)):
            mu_strip = mu[ch]
            mu_strip = (mu_strip - mu_strip.min()) / (mu_strip.max() - mu_strip.min() + 1e-6)
            mu_strip_img = self._stack_horizontal(mu_strip).detach().cpu().numpy()
            wandb.log({f"{prefix}/mu_tstrip_c{ch}": wandb.Image(mu_strip_img, caption=f"mu channel {ch} over time")}, step=step)

            z_strip = z[ch]
            z_strip = (z_strip - z_strip.min()) / (z_strip.max() - z_strip.min() + 1e-6)
            z_strip_img = self._stack_horizontal(z_strip).detach().cpu().numpy()
            wandb.log({f"{prefix}/z_tstrip_c{ch}": wandb.Image(z_strip_img, caption=f"z channel {ch} over time")}, step=step)

    def _latent_video_grid(self, mu: torch.Tensor, channels=None, fps: int = 4,
                           use_quantiles: bool = True, q_lo: float = 0.5, q_hi: float = 99.5,
                           rgb: bool = False):
        mu = mu.detach().float().cpu()
        C, n, H, W = mu.shape
        if channels is None:
            k = min(16, C)
            channels = list(range(k))
        else:
            channels = list(channels)[:16]
            k = len(channels)
        cols = 8 if k > 8 else k
        rows = int(math.ceil(k / cols))
        sel = mu[channels]
        flat = sel.reshape(k, -1)
        if use_quantiles:
            lo = torch.quantile(flat, q_lo / 100.0, dim=1)
            hi = torch.quantile(flat, q_hi / 100.0, dim=1)
        else:
            lo = flat.min(dim=1).values
            hi = flat.max(dim=1).values
        denom = (hi - lo).clamp_min(1e-6)
        frames = []
        for t in range(n):
            row_tiles = []
            for r in range(rows):
                row_imgs = []
                for c in range(cols):
                    idx = r * cols + c
                    if idx < k:
                        m = sel[idx, t]
                        g = (m - lo[idx]) / denom[idx]
                        g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8)
                    else:
                        g = torch.zeros((H, W), dtype=torch.uint8)
                    row_imgs.append(g)
                row_tiles.append(torch.cat(row_imgs, dim=-1))
            grid = torch.cat(row_tiles, dim=-2).unsqueeze(0)
            frames.append(grid)
        vid = torch.stack(frames, dim=0)
        vid = vid.repeat(1, 3, 1, 1)
        return wandb.Video(vid.numpy(), fps=int(fps), format="mp4")

    # ---------- shared forward + loss (stage-aware, ragged) ----------
    # (training path unchanged)
    def compute_loss(self, batch_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self.accelerator.device
        xs = []
        for x in batch_list:
            x = x.to(device=device, dtype=torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            assert x.dim() == 4 and x.shape[0] == 2, f"Each sample must be [2,T,H,W], got {tuple(x.shape)}"
            xs.append(x)

        if self.mode == "pretrain_2d":
            xhats, mus, logvs, zs = self.model(xs)

            img_vals, lpips_vals, kl_vals = [], [], []
            for x, xhat, mu, logv, z in zip(xs, xhats, mus, logvs, zs):
                xhatv = xhat.squeeze(0)
                mu, logv = mu.squeeze(0), logv.squeeze(0)
                z = z.squeeze(0)

                if self.cfg["loss"][self.mode]["complex_l1"]:
                    img_vals.append(self._charb_l1(xhatv, x))
                else:
                    img_vals.append(self._charb_l1_mag(xhatv, x))

                if self.use_lpips:
                    lpips_vals.append(self._lpips_loss_item(x, xhatv, mode="pretrain_2d"))
                kl_vals.append(self._kl_gaussian_channels(mu, logv))

            img = torch.stack(img_vals).mean()
            lpips_loss = torch.stack(lpips_vals).mean() if lpips_vals else torch.zeros((), device=device)
            kl  = torch.stack(kl_vals).mean()

            w = self.current_w
            total = (w["img"] * img + w["lpips"] * lpips_loss + w["kl"] * kl)

            return {"total": total, "img": img, "lpips": lpips_loss, "kl": kl,
                    "xs": xs, "xhats": [xhat.squeeze(0).detach() for xhat in xhats]}

        else:
            xhats, mus, logvs, zs = self.model(xs)

            img_vals, tvz_vals, kl_vals, lpips_vals = [], [], [], []
            for x, xhat, mu, logv, z in zip(xs, xhats, mus, logvs, zs):
                xhatv = xhat.squeeze(0)
                mu, logv = mu.squeeze(0), logv.squeeze(0)
                z = z.squeeze(0)

                if self.cfg["loss"][self.mode]["complex_l1"]:
                    img_vals.append(self._charb_l1(xhatv, x))
                else:
                    img_vals.append(self._charb_l1_mag(xhatv, x))

                tvz_vals.append(self._tv3d(z))
                kl_vals.append(self._kl_gaussian_channels(mu, logv))
                if self.use_lpips:
                    lpips_vals.append(self._lpips_loss_item(x, xhatv, mode="videos"))

            img = torch.stack(img_vals).mean()
            tvz = torch.stack(tvz_vals).mean()
            kl  = torch.stack(kl_vals).mean()
            lpips_loss = torch.stack(lpips_vals).mean() if lpips_vals else torch.zeros((), device=device)

            w = self.current_w
            total = (w["img"] * img + w["lpips"] * lpips_loss +
                    w["tvz"] * tvz + w["kl"] * kl)

            return {"total": total, "img": img, "lpips": lpips_loss,
                    "tvz": tvz, "kl": kl,
                    "xs": xs, "xhats": [xhat.squeeze(0) for xhat in xhats]}

    def _losses_are_finite(self, d: dict) -> bool:
        ok = True
        for k, v in d.items():
            if k in ("xs", "xhats"):
                continue
            if torch.is_tensor(v):
                if v.ndim == 0 and not torch.isfinite(v):
                    ok = False
                    break
        return ok

    # ---------- training loop (unchanged) ----------
    def train(self):
        log_cfg = self.cfg["logging"]
        opt_cfg = self.cfg["optim"]

        self.accelerator.print(f"Starting training (mode={self.mode}, float32, ragged batches)…")

        pbar = tqdm(
            total=self.total_steps - self.start_step,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            desc="train",
            leave=True,
        )
        last = time.perf_counter()

        train_iter = iter(self.train_dl)

        while self.global_step < self.total_steps:
            try:
                batch_list = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch_list = next(train_iter)

            with self.accelerator.accumulate(self.model):
                losses = self.compute_loss(batch_list)
                torch.cuda.empty_cache()
                if not self._losses_are_finite(losses):
                    if self.accelerator.is_main_process:
                        print(f"[warn] non-finite loss component at step {self.global_step}; skipping iteration.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss = losses["total"]
                if not torch.isfinite(loss):
                    if self.accelerator.is_main_process:
                        print(f"[warn] non-finite loss at step {self.global_step}; masking this step.")
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # if self.grad_clip and self.grad_clip > 0:
                        # self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    now = time.perf_counter()
                    dt = now - last
                    last = now

                    local_count = len(batch_list)

                    avg = self._avg_losses_across_gpus(losses, count=local_count)

                    if self.accelerator.is_main_process:
                        pbar.update(1)
                        pbar.set_postfix(
                            loss=f"{avg.get('total', float(loss.detach().cpu())):.4f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.3e}",
                            sec_it=f"{dt:.3f}",
                        )

                    if self.accelerator.is_main_process and self.global_step % log_cfg["log_every_steps"] == 0:
                        scalars = {f"train/{k}": v for k, v in avg.items()}
                        scalars["lr"] = self.optimizer.param_groups[0]["lr"]
                        scalars.update({f"w/{k}": float(v) for k, v in self.current_w.items()})
                        wandb.log(scalars, step=self.global_step)

                    if (self.global_step % log_cfg["val_every_steps"] == 0) and self.global_step > self.start_step:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[val] step {self.global_step}")
                        self.validate()
                        self.accelerator.wait_for_everyone()

                    if self.global_step == 2000:
                        if self.accelerator.is_main_process:
                            print("[scale] calibrating latent scale from train patches…")
                        self.calibrate_latent_scale(num_batches=16, use_train=True)
                        self.accelerator.wait_for_everyone()

                    if (self.global_step % log_cfg["save_every_steps"] == 0) and self.global_step > self.start_step:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[ckpt] step {self.global_step}")
                        self.save_checkpoint()
                        self.accelerator.wait_for_everyone()

                    self.global_step += 1
                    self.current_w = self.loss_sched.weights_for_step(self.global_step)

                    if self.global_step >= self.total_steps:
                        break
                    # if self.global_step >= self.cfg.get("debug_steps", 200000):
                        # break

        if self.accelerator.is_main_process:
            pbar.close()
            self.accelerator.print("Training complete.")

    @torch.no_grad()
    def calibrate_latent_scale(self, num_batches: int = 16, use_train: bool = True):
        self.model.eval()
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)
        C = int(unwrapped.z_dim)

        sum_c   = torch.zeros(C, device=device, dtype=torch.float32)
        sumsq_c = torch.zeros(C, device=device, dtype=torch.float32)
        cnt_c   = torch.zeros(C, device=device, dtype=torch.float32)

        loader = self.train_dl if use_train else self.val_dl
        it = iter(loader)
        for _ in range(num_batches):
            try:
                batch_list = next(it)
            except StopIteration:
                break

            for x in batch_list:
                x = x.to(device=device, dtype=torch.float32)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if x.dim() == 4:
                    x = x.unsqueeze(0)

                mu = unwrapped.encode_raw_mu(x)
                m  = mu.sum(dim=(0, 2, 3, 4))
                ss = (mu * mu).sum(dim=(0, 2, 3, 4))
                nvox = float(mu.shape[0] * mu.shape[2] * mu.shape[3] * mu.shape[4])

                sum_c   += m
                sumsq_c += ss
                cnt_c   += nvox

        world = self.accelerator.num_processes
        def _gather_and_sum(v):
            g = self.accelerator.gather_for_metrics(v)
            return g.view(world, -1).sum(dim=0)

        sum_all   = _gather_and_sum(sum_c)
        sumsq_all = _gather_and_sum(sumsq_c)
        cnt_all   = _gather_and_sum(cnt_c)

        mean = sum_all / cnt_all.clamp_min(1.0)
        var  = (sumsq_all / cnt_all.clamp_min(1.0)) - mean * mean
        std  = var.clamp_min(1e-8).sqrt().clamp_min(1e-3)

        unwrapped.set_scale(mean.detach().cpu(), std.detach().cpu())

        if self.accelerator.is_main_process:
            wandb.log(
                {
                    "calib/mean_abs": float(mean.abs().mean().cpu()),
                    "calib/std_mean": float(std.mean().cpu()),
                },
                step=self.global_step,
            )
            self.accelerator.print("[scale] calibrated per-channel μ mean/std and set via set_scale().")

        self.model.train()

    # ---------- VALIDATION: patchify full-res, run VAE on patches, depatchify ----------
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        # Keep key names unchanged
        keys = ["total","img","k","kl","lpips"] if self.mode=="pretrain_2d" else ["total","img","k","lpips","tvz","kl"]

        device = self.accelerator.device
        meters_sum = {k: torch.zeros((), device=device, dtype=torch.float32) for k in keys}
        items_cnt  = torch.zeros((), device=device, dtype=torch.float32)

        sse_mag_rank = torch.zeros((), device=device, dtype=torch.float32)
        cnt_mag_rank = torch.zeros((), device=device, dtype=torch.float32)
        # For complex SNR aggregation across the whole validation set
        energy_sig_rank = torch.zeros((), device=device, dtype=torch.float32)
        energy_err_rank = torch.zeros((), device=device, dtype=torch.float32)

        logged_media    = False
        logged_latents  = False
        logged_patchviz = False

        val_batches = int(self.cfg["logging"].get("val_num_batches", 8))
        val_iter = iter(self.val_dl)

        # Helper: run model on a list of patches (each [2,t,h,w]) in micro-batches
        def run_patches_through_vae(patches_list: List[torch.Tensor]):
            xhat_list, mu_list, logv_list, z_list = [], [], [], []
            bs = max(1, self.val_patch_bs)
            for i in range(0, len(patches_list), bs):
                chunk = patches_list[i:i+bs]                   # list of [2,t,h,w]
                xhats, mus, logvs, zs = self.model(chunk)     # lists aligned to chunk
                for xh, mu, lv, z in zip(xhats, mus, logvs, zs):
                    xhat_list.append(xh.squeeze(0))           # [2,t,h,w]
                    mu_list.append(mu.squeeze(0))             # [Cz,n,H',W']
                    logv_list.append(lv.squeeze(0))
                    z_list.append(z.squeeze(0))
            return xhat_list, mu_list, logv_list, z_list

        for _ in range(val_batches):
            try:
                batch_list = next(val_iter)
            except StopIteration:
                break

            for x in batch_list:
                # x: full-res tensor [2,T,H,W]
                x = x.to(device=device, dtype=torch.float32)
                _, T, H, W = x.shape

                # -------- patchify along (T,H,W) using colleague logic --------
                patch_t = self.val_patch_t if T > 1 else 1  # pretrain_2d uses 1
                patch_h = self.val_patch_h
                patch_w = self.val_patch_w
                patch_size = (patch_t, patch_h, patch_w)

                # Real/imag patchify separately, stack to [N,2,t,h,w]
                r_patches, strides = patchify(x[0], patch_size)  # [N,t,h,w]
                i_patches, _       = patchify(x[1], patch_size)  # [N,t,h,w]
                Np = int(r_patches.shape[0])

                patches_list = [ torch.stack((r_patches[n], i_patches[n]), dim=0) for n in range(Np) ]  # list of [2,t,h,w]

                # -------- run VAE on each patch (batched) --------
                xhat_list, mu_list, logv_list, z_list = run_patches_through_vae(patches_list)

                # -------- reconstruct full-resolution video via depatchify --------
                # per-channel overlap-add
                r_rec_patches = torch.stack([xh[0] for xh in xhat_list], dim=0)  # [N,t,h,w]
                i_rec_patches = torch.stack([xh[1] for xh in xhat_list], dim=0)  # [N,t,h,w]

                xhat_r = depatchify(r_rec_patches, (T, H, W), patch_size, strides)  # [T,H,W]
                xhat_i = depatchify(i_rec_patches, (T, H, W), patch_size, strides)  # [T,H,W]
                xhat   = torch.stack((xhat_r, xhat_i), dim=0)                        # [2,T,H,W]

                # -------- compute reconstruction metrics on FULL video --------
                if self.cfg["loss"][self.mode]["complex_l1"]:
                    img_loss = self._charb_l1(xhat, x)
                    k_loss   = self._charb_l1(self._fft2c(xhat), self._fft2c(x))
                else:
                    img_loss = self._charb_l1_mag(xhat, x)
                    k_loss   = self._charb_l1_mag(self._fft2c(xhat), self._fft2c(x))

                lpips_loss = self._lpips_loss_item(x, xhat, mode=("pretrain_2d" if T==1 else "videos"))

                # -------- latent losses aggregated ACROSS ALL PATCHES --------
                if self.mode == "videos" and len(z_list) > 0:
                    tvz_vals = [self._tv3d(zp) for zp in z_list]
                    tvz = torch.stack(tvz_vals).mean()
                else:
                    tvz = torch.zeros((), device=device, dtype=torch.float32)

                kl_vals = [self._kl_gaussian_channels(mu, lv) for (mu, lv) in zip(mu_list, logv_list)]
                kl = torch.stack(kl_vals).mean() if kl_vals else torch.zeros((), device=device, dtype=torch.float32)

                # Compose total with current weights (do NOT add k into total)
                w = self.current_w
                if self.mode == "pretrain_2d":
                    total = (w["img"] * img_loss + w["lpips"] * lpips_loss + w["kl"] * kl)
                else:
                    total = (w["img"] * img_loss + w["lpips"] * lpips_loss + w["tvz"] * tvz + w["kl"] * kl)

                # -------- accumulate meters (per video) --------
                met = {"total": total, "img": img_loss, "k": k_loss, "kl": kl, "lpips": lpips_loss}
                if self.mode == "videos":
                    met["tvz"] = tvz
                for k in keys:
                    meters_sum[k] += met[k].detach().to(device=device, dtype=torch.float32)
                items_cnt += 1.0

                # -------- PSNR accum on magnitude over all frames/pixels --------
                xm  = self._complex_mag(x)
                xhm = self._complex_mag(xhat)
                diff = xhm - xm
                sse_mag_rank += (diff * diff).sum().to(device)
                cnt_mag_rank += float(xm.numel())

                # -------- SNR accum on complex signal (real/imag jointly) --------
                # SNR(xhat,x) [dB] = 10*log10( ||x||^2 / ||xhat-x||^2 )
                # Here ||·|| is the ℓ2 norm over the complex field => sum over real/imag channels.
                err = (xhat - x)
                energy_sig_rank += (x * x).sum().to(device)
                energy_err_rank += (err * err).sum().to(device)

                # -------- One-time visuals --------
                if self.accelerator.is_main_process and not logged_media:
                    if T == 1:
                        xm1  = xm[:, 0]
                        xhm1 = xhm[:, 0]
                        xm_n  = (xm1 - xm1.min()) / (xm1.max() - xm1.min() + 1e-6)
                        xhm_n = (xhm1 - xhm1.min()) / (xhm1.max() - xhm1.min() + 1e-6)
                        wandb.log({
                            "vis/img_mag":   wandb.Image(xm_n.cpu().numpy(),  caption="GT magnitude"),
                            "vis/recon_mag": wandb.Image(xhm_n.cpu().numpy(), caption="Recon magnitude"),
                        }, step=self.global_step)
                    else:
                        media = {
                            "vis/video_gt":    self._to_wandb_video_one(xm),
                            "vis/video_recon": self._to_wandb_video_one(xhm),
                        }
                        wandb.log(media, step=self.global_step)
                    logged_media = True

                # -------- Patchify/Depatchify sanity visualization (main only, once) --------
                if self.accelerator.is_main_process and not logged_patchviz:
                    try:
                        # Use the SAME sample x processed above
                        _, T_v, H_v, W_v = x.shape
                        t_mid = T_v // 2

                        # Prepare magnitude volume in (H, W, T) for patch ops
                        xm_v   = self._complex_mag(x).squeeze(0)          # [T,H,W]
                        data3d = xm_v.permute(1, 2, 0).contiguous()       # [H,W,T]

                        # Patch params (spatial 80×80, temporal 11)
                        P0, P1, P2 = 80, 80, 11
                        patch_size_viz = (P0, P1, P2)

                        # Compute strides and number of patches
                        strides_viz, _, n_list = compute_strides_and_N(data3d.shape, patch_size_viz)
                        S0, S1, S2 = strides_viz
                        n0, n1, n2 = n_list

                        # Identity patchify/depatchify on GT (no VAE)
                        patches_viz, strides_rt = patchify(data3d, patch_size_viz)      # [N, P0, P1, P2]
                        id_recon = depatchify(patches_viz, data_shape=data3d.shape,
                                            patch_size=patch_size_viz, strides=strides_rt)  # [H,W,T]

                        # Panels at mid-frame (0..255)
                        def _to_uint8_2d(m_2d: torch.Tensor) -> np.ndarray:
                            return self._frame_to_uint8(m_2d.unsqueeze(0))

                        gt_mid  = _to_uint8_2d(data3d[:, :, t_mid])
                        id_mid  = _to_uint8_2d(id_recon[:, :, t_mid])
                        diff    = (torch.abs(id_recon[:, :, t_mid] - data3d[:, :, t_mid]) * 4.0).clamp(0, 1.0)
                        diff_u8 = (diff.detach().cpu().numpy() * 255.0).round().astype(np.uint8)

                        # Overlay grid on GT
                        grid = np.stack([gt_mid, gt_mid, gt_mid], axis=-1)      # [H,W,3] RGB
                        # Horizontal (red)
                        for i_ in range(n0):
                            y0 = i_ * S0
                            y1 = min(y0 + P0, H_v) - 1
                            if 0 <= y0 < H_v: grid[y0, :, :] = [255, 0, 0]
                            if 0 <= y1 < H_v: grid[y1, :, :] = [255, 0, 0]
                        # Vertical (green)
                        for j_ in range(n1):
                            x0 = j_ * S1
                            x1 = min(x0 + P1, W_v) - 1
                            if 0 <= x0 < W_v: grid[:, x0, :] = [0, 255, 0]
                            if 0 <= x1 < W_v: grid[:, x1, :] = [0, 255, 0]

                        panel = np.concatenate([
                            np.stack([gt_mid]*3, axis=-1),
                            grid,
                            np.stack([id_mid]*3, axis=-1),
                            np.stack([diff_u8]*3, axis=-1),
                        ], axis=1)  # [H, 4*W, 3]

                        wandb.log(
                            {"vis/patchify_depatchify_debug": wandb.Image(panel,
                            caption=f"GT | GT+grid | Identity(repatch) | |GT−ID|×4 @ t={t_mid}")},
                            step=self.global_step
                        )
                        logged_patchviz = True
                    except Exception as e:
                        self.accelerator.print(f"[viz] patchify/depatchify panel skipped: {e}")

                # -------- One-time latent diagnostics (use first patch’s mu/z) --------
                if self.accelerator.is_main_process and not logged_latents and len(mu_list) > 0:
                    mu0, logv0, z0 = mu_list[0], logv_list[0], z_list[0]      # [Cz,n,H',W']
                    self._log_latent_images(mu0, z0, step=self.global_step, prefix="vis/latent")
                    if self.mode == "videos":
                        fps = int(self.cfg["logging"].get("latent_grid_fps", 4))
                        wandb.log({"vis/latent_video/mu_grid": self._latent_video_grid(mu0, fps=fps)}, step=self.global_step)

                    # Stats across ALL patches of this video
                    mu_cat   = torch.cat(mu_list,  dim=1)   # [Cz, sum_n, H', W']
                    logv_cat = torch.cat(logv_list, dim=1)
                    stats = self._latent_stats(mu_cat, logv_cat)
                    wandb.log({f"val/{k}": (v.detach().float().item() if torch.is_tensor(v) else float(v))
                            for k, v in stats.items()}, step=self.global_step)

                    # Radial k-space band errors on full recon
                    band_errs = self._radial_band_errors_one(x, xhat)
                    wandb.log({f"val/{k}": v.detach().float().item() for k, v in band_errs.items()},
                            step=self.global_step)

                    # Temporal jerk on latent (use one patch for visualization)
                    if z0.shape[1] >= 3:
                        jerk = (z0[:, 2:] - 2*z0[:, 1:-1] + z0[:, :-2]).abs().mean()
                        wandb.log({"val/latent_temporal_jerk": jerk.detach().float().item()}, step=self.global_step)

                    logged_latents = True

        # ---------------- cross-GPU reductions ----------------
        g_sums = {k: self.accelerator.gather_for_metrics(v) for k, v in meters_sum.items()}
        g_cnts = self.accelerator.gather_for_metrics(items_cnt)

        global_means = {}
        denom = g_cnts.sum().clamp_min(1.0)
        for k, vec in g_sums.items():
            global_means[k] = float(vec.sum().item() / float(denom))

        sse_all = self.accelerator.gather_for_metrics(sse_mag_rank)
        cnt_all = self.accelerator.gather_for_metrics(cnt_mag_rank)
        sse_sum = sse_all.sum()
        cnt_sum = cnt_all.sum().clamp_min(1.0)
        mse_mag = sse_sum / cnt_sum
        max_I   = torch.sqrt(torch.tensor(2.0, device=device))
        psnr    = 10.0 * torch.log10((max_I * max_I) / mse_mag.clamp_min(1e-12))
        global_means["psnr_mag"] = float(psnr.detach().cpu().item())

        # -------- finalize complex SNR over entire validation set --------
        sig_all = self.accelerator.gather_for_metrics(energy_sig_rank)
        err_all = self.accelerator.gather_for_metrics(energy_err_rank)
        sig_sum = sig_all.sum().clamp_min(1e-12)
        err_sum = err_all.sum().clamp_min(1e-12)
        snr_db  = 10.0 * torch.log10(sig_sum / err_sum)
        global_means["snr_complex"] = float(snr_db.detach().cpu().item())

        # ---------------- logging ----------------
        if self.accelerator.is_main_process:
            wandb.log({f"val/{k}": v for k, v in global_means.items()}, step=self.global_step)
            self.accelerator.print(
                f"Val @ step {self.global_step}: " +
                " ".join([f"{k}={v:.4f}" for k, v in global_means.items()])
            )

        self.model.train()

    def _avg_losses_across_gpus(self, stats: dict, count: int) -> dict:
        keys, vals = [], []
        for k, v in stats.items():
            if k in ("xs", "xhats"):
                continue
            if torch.is_tensor(v) and v.ndim == 0:
                keys.append(k)
                vals.append(v.detach().to(self.accelerator.device, dtype=torch.float32))
        if not keys:
            return {}
        vec = torch.stack(vals)
        cnt = torch.tensor([float(count)], device=self.accelerator.device, dtype=torch.float32)
        g_vec = self.accelerator.gather_for_metrics(vec)
        g_cnt = self.accelerator.gather_for_metrics(cnt)
        world = self.accelerator.num_processes
        g_vec = g_vec.view(world, -1)
        g_cnt = g_cnt.view(world, 1)
        sums   = (g_vec * g_cnt).sum(dim=0)
        counts = g_cnt.sum(dim=0).clamp_min(1.0)
        means  = sums / counts
        return {k: float(means[i].item()) for i, k in enumerate(keys)}

    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        save_dir = os.path.join(self.cfg["logging"]["out_dir"], f"step_{self.global_step:07d}")
        os.makedirs(save_dir, exist_ok=True)

        state = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "global_step": self.global_step,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(save_dir, "state.pt")
        self.accelerator.save(state, path)
        if self.accelerator.is_main_process:
            wandb.save(path)
            self.accelerator.print(f"Saved checkpoint: {save_dir}")


# -------------------- helpers --------------------

def build_dataloader(ds_cfg: Dict[str, Any], dl_cfg: Dict[str, Any], is_train: bool) -> DataLoader:
    name = ds_cfg["name"]; args = ds_cfg.get("args", {})
    if name not in ("CINEDataset"):
        raise ValueError("Adjust build_dataloader: expected CINEDataset.")
    dataset = CINEDataset(**args)  # returns [2,1,H,W] or [2,L,H,W]
    bsz = dl_cfg["train_batch_size"] if is_train else dl_cfg["val_batch_size"]
    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=dl_cfg.get("shuffle", True) if is_train else False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=is_train,
        collate_fn=ragged_collate,
    )

def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg['train_dataset']['args']['stage_mode'] = cfg['stages']['mode']
    cfg['val_dataset']['args']['stage_mode'] = cfg['stages']['mode']
    cfg['test_dataset']['args']['stage_mode'] = cfg['stages']['mode']

    cfg['logging']['run_name'] = cfg['logging']['run_name'] + "_" + cfg['stages']['mode']
    torch.manual_seed(42)

    train_dl = build_dataloader(cfg["train_dataset"], cfg["dataloader"], is_train=True)
    val_dl   = build_dataloader(cfg["val_dataset"],   cfg["dataloader"], is_train=False)

    CardiacVAE = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = CardiacVAE(**cfg["model"]["args"])

    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))

    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=strict_load)
        print(f"[VAE] loaded pretrained from {pretrained_path}")
        if not strict_load:
            print(f"[VAE] missing={len(missing)} unexpected={len(unexpected)}")

    # Print total number of model parameters
    print(f"[VAE] total number of model parameters: {sum(p.numel() for p in model.parameters())}")

    trainer = CardiacVAETrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae.yaml")
    args = parser.parse_args()
    main(args.config)
