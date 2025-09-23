#!/usr/bin/env python3
import os, sys, time, importlib, math, yaml
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

# Add cwd to path
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import wandb

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# EMA
from utils.ema import Ema

# UniPC Flow Matching sampler (expects model(x, t, **extra_args))
from CardiacFM.sampler.flow_match_uni_pc import sample_unipc
from data.deg import MRIDeg


# ==================== time utilities (Flux-style reparam) ====================

def flux_time_shift(t: torch.Tensor, mu=1.15, sigma: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = torch.clamp(t, min=1e-5, max=1.0)
    mu_t = torch.as_tensor(mu, device=t.device, dtype=torch.float32)
    emu = torch.exp(mu_t)
    return emu / (emu + (1.0 / t - 1.0).pow(sigma))

def get_flux_sigmas_from_mu(n_steps: int, mu, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t_f32 = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device, dtype=torch.float32)
    sig = flux_time_shift(t_f32, mu=mu).to(dtype)
    return sig

def calculate_flux_mu(context_length: int,
                      x1: float = 256, y1: float = 0.5,
                      x2: float = 4096, y2: float = 1.15,
                      exp_max: float = 7.0) -> float:
    k = (y2 - y1) / max(1.0, (x2 - x1))
    b = y1 - k * x1
    mu = k * float(context_length) + b
    return float(min(mu, math.log(exp_max)))


# ==================== dataset + collate ====================

def ragged_collate(batch):
    # keep as list; clips may have varying L
    return batch

def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def try_import_dataset(name: str):
    if name == "CINEPixelDataset":
        mod = importlib.import_module("data.cine_dataset_pixel")
        return getattr(mod, "CINEPixelDataset")
    elif name in ("CINEDataset", "CINEFlowMatchDataset"):
        mod = importlib.import_module("data.cine_dataset")
        return getattr(mod, "CINEDataset")
    else:
        raise ValueError(f"Unknown dataset '{name}'.")


# ==================== small viz helpers ====================
def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

# ==================== 2D spatial patchify (val) with overlap-add ====================

def temporal_starts_full_coverage(T_total: int, T_win: int, prefer_stride: int) -> List[int]:
    if T_total <= T_win: return [0]
    s = max(1, int(prefer_stride))
    starts = list(range(0, T_total - T_win + 1, s))
    if starts[-1] + T_win < T_total:
        n = len(starts) + 1
        s = max(1, math.floor((T_total - T_win) / (n - 1)))
        starts = [i * s for i in range(n - 1)]
        last = T_total - T_win
        if starts[-1] != last:
            starts.append(last)
    return starts

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


# ==================== FFT helpers for [2,T,H,W] ====================

def fft2c(x_2thw: torch.Tensor) -> torch.Tensor:
    xr, xi = x_2thw[0], x_2thw[1]
    xc = torch.complex(xr, xi)               # [T,H,W]
    k  = torch.fft.fft2(xc, norm="ortho")
    kc = torch.fft.fftshift(k, dim=(-2, -1))
    return torch.stack((kc.real, kc.imag), dim=0)  # [2,T,H,W]

def ifft2c(kc_2thw: torch.Tensor) -> torch.Tensor:
    kr, ki = kc_2thw[0], kc_2thw[1]
    kc = torch.complex(kr, ki)
    k  = torch.fft.ifftshift(kc, dim=(-2, -1))
    x  = torch.fft.ifft2(k, norm="ortho")
    return torch.stack((x.real, x.imag), dim=0)    # [2,T,H,W]


# ==================== Trainer (PIXEL space) ====================

class PixelFMTrainer:
    """
    Train a velocity model directly on pixel clips x ~ [2, L, 80, 80] (bf16).
    Validation:
      • Unconditional: sample pixel patches via UniPC, log grid
      • Conditional (inverse): Euler(x0)+DC in pixel space (no VAE anywhere)
    """
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        cfg['logging']['out_dir'] = os.path.join(cfg['logging']['out_dir'], "flowmatch_pixels")
        self.cfg = cfg
        self.model = model
        self.model = torch.compile(self.model, fullgraph=False)

        self.t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))

        proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)
        self.accelerator = Accelerator(project_config=proj_cfg, kwargs_handlers=[ddp_kwargs], mixed_precision="bf16")

        opt_cfg = cfg["optim"]
        self.total_steps = int(opt_cfg["total_steps"])
        self.accum_steps = int(opt_cfg.get("accum_steps", 1))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            weight_decay=opt_cfg["weight_decay"],
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
        resume_state = None
        resume_path = cfg["model"].get("load_state_dict_from", None)
        resume_flag = bool(cfg["model"].get("resume", False))
        if resume_path and resume_flag and os.path.isfile(resume_path):
            try:
                resume_state = torch.load(resume_path, map_location="cpu")
                if "optimizer" in resume_state:
                    self.optimizer.load_state_dict(resume_state["optimizer"])
                if self.scheduler is not None and "scheduler" in resume_state:
                    self.scheduler.load_state_dict(resume_state["scheduler"])
                self.global_step = int(resume_state.get("global_step", 0))
                print(f"[resume] loaded opt/sched and step={self.global_step} from {resume_path}")
            except Exception as e:
                print(f"[resume] failed to load optimizer/scheduler from {resume_path}: {e}")

        self.start_step = self.global_step

        if self.scheduler is not None:
            (self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler)
        else:
            (self.model, self.optimizer, self.train_dl, self.val_dl) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema = Ema(unwrapped, decay=float(self.cfg["optim"].get("ema_decay", 0.999)))

        self.grad_clip = float(opt_cfg.get("grad_clip", 0.0))

        # wandb dirs
        wdir  = self.cfg["logging"].get("wandb_dir", None)
        wcache = self.cfg["logging"].get("wandb_cache_dir", None)
        if wdir:   os.environ["WANDB_DIR"] = str(wdir)
        if wcache: os.environ["WANDB_CACHE_DIR"] = str(wcache)

        if self.accelerator.is_main_process:
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"].get("run_name", "pixel_fm_bf16"),
                config=cfg,
                dir=wdir,
            )

        # -------- Validation patch config (pixel) --------
        vcfg = self.cfg.get("validation", {})
        self.val_patch_h  = int(vcfg.get("patch_h", 80))
        self.val_patch_w  = int(vcfg.get("patch_w", 80))
        self.val_patch_t  = int(vcfg.get("patch_t", 7))
        self.val_patch_bs = int(vcfg.get("patch_batch", 64))  # for batching patches through the net (uncond)

    # -------- Rectified-Flow loss on PIXELS (bf16 forward, fp32 loss) --------
    def _rectified_flow_loss_pixels(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        X: [N, 2, L, H, W] (bf16)
        x0=X, x1~N(0,1), t~U(0,1)
        x_t=(1-t)X + t x1, target u*=x1 - X
        """
        device = X.device
        N = X.shape[0]
        noise  = torch.randn_like(X)               # bf16
        sigmas = torch.rand((N,), device=device, dtype=torch.bfloat16)

        t_b    = sigmas.view(N, *([1]*(X.ndim-1)))
        x_t    = (1.0 - t_b) * X + t_b * noise     # bf16
        target = (noise - X)                       # bf16

        t_inp  = (sigmas * self.t_scale)                # bf16

        pred = self.model(x_t, t_inp)              # bf16 velocity
        if self.accelerator.is_main_process:
            data_max = X.max()
            data_min = X.min()
            pred_norm = pred[0].norm()
            target_norm = target[0].norm()
            print(f't_scale: {self.t_scale}, data_max: {data_max:.4f}, data_min: {data_min:.4f}, pred_norm: {pred_norm:.4f}, target_norm: {target_norm:.4f}, t: {sigmas[0]:.4f}')

        mse  = torch.nn.functional.mse_loss(pred.float(), target.float())
        return {"total": mse, "mse": mse}

    def _fm_loss(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._rectified_flow_loss_pixels(X)

    # -------- training step (pixels) --------
    def compute_loss(self, batch_list: List[Any]) -> Dict[str, torch.Tensor]:
        """
        TRAIN batches: list of pixel clips [2, L, 80, 80] (L odd; may vary).
        We stack only those with identical shapes; otherwise compute per-item and average.
        """
        device = self.accelerator.device

        X = torch.stack(batch_list, dim=0)  # [B,2,L,H,W]
        losses = self._fm_loss(X)
        l = losses["total"]
        m = losses["mse"]

        total_loss = l
        total_mse  = m
        return {"total": total_loss, "mse": total_mse}

    # -------- training loop --------
    def train(self):
        log_cfg = self.cfg["logging"]
        self.accelerator.print("Starting Pixel Flow Matching training (bf16)…")

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
                loss = losses["total"]
                if not torch.isfinite(loss):
                    if self.accelerator.is_main_process:
                        print(f"[warn] non-finite loss at step {self.global_step}; masking to 0.")
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if self.grad_clip and self.grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.accelerator.unwrap_model(self.model))
                    if self.scheduler is not None:
                        self.scheduler.step()

                    now = time.perf_counter()
                    dt = now - last
                    last = now

                    if self.accelerator.is_main_process:
                        pbar.update(1)
                        pbar.set_postfix(
                            loss=f"{float(loss.detach().cpu()):.4f}",
                            lr=f"{self.optimizer.param_groups[0]['lr']:.3e}",
                            sec_it=f"{dt:.3f}",
                        )

                    if self.accelerator.is_main_process and (self.global_step % log_cfg["log_every_steps"] == 0):
                        scalars = {f"train/{k}": float(v.detach().cpu()) for k, v in losses.items() if torch.is_tensor(v)}
                        scalars["lr"] = self.optimizer.param_groups[0]["lr"]
                        wandb.log(scalars, step=self.global_step)

                    if (self.global_step % log_cfg["val_every_steps"] == 0) and self.global_step > 0:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[val] step {self.global_step}")

                        self.validate_fixed_t_x0_from_data((0.01, 0.10, 0.25, 0.50, 0.75, 1.00))
                        self.validate_uncond()

                        if self.global_step > 100000:
                            self.validate_inverse_dc()
                        
                        self.accelerator.wait_for_everyone()

                    if (self.global_step % log_cfg["save_every_steps"] == 0) and self.global_step > 0:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[ckpt] step {self.global_step}")
                        self.save_checkpoint()
                        self.accelerator.wait_for_everyone()

                    self.global_step += 1
                    if self.global_step >= self.total_steps:
                        break

        if self.accelerator.is_main_process:
            pbar.close()
            self.accelerator.print("Training complete.")

    @torch.no_grad()
    def validate_fixed_t_x0_from_data(self, t_values=(0.01, 0.10, 0.25, 0.50, 0.75, 0.999)):
        """
        RAW weights. Take true pixel patches from the val set, build x_t,
        predict u(x_t, t), compute x0_pred = x_t - t * u, and log a video per t.
        Also:
        • 'oracle' check with u* = x1 - x0 to verify math/broadcasting
        • compare 'scaled t' (×t_scale) vs 'unscaled' t to detect t-scale bugs
        """
        self.model.eval()
        device = self.accelerator.device

        # --- use the SAME val patch geometry you set in __init__ ---
        pt = int(self.val_patch_t)
        ph = int(self.val_patch_h)
        pw = int(self.val_patch_w)
        N  = int(self.cfg.get("validation", {}).get("num_uncond_videos", 8))

        # -------- fetch one validation volume [2,T,H,W] --------
        try:
            val_it = iter(self.val_dl)
            batch_list = next(val_it)
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[validate_fixed_t_x0_from_data] empty val loader")
            self.model.train()
            return

        x_true = None
        for item in batch_list:
            x_true = item
            break

        x_true = x_true.to(device=device, dtype=torch.float32)
        if x_true.dim() == 3:
            x_true = x_true.unsqueeze(1)  # [2,1,H,W] -> [2,T,H,W]
        _, T, H, W = x_true.shape

        # -------- deterministic center clip of length pt (wrap if needed) --------
        if T >= pt:
            t0 = (T - pt) // 2
            x_clip = x_true[:, t0:t0 + pt]  # [2,pt,H,W]
        else:
            reps = (pt + T - 1) // T
            x_clip = x_true.repeat(1, reps, 1, 1)[:, :pt]  # [2,pt,H,W]

        # -------- spatial patches, stride == patch (no overlap) --------
        sh, sw = ph, pw
        coords = spatial_coords(H, W, ph, pw, sh, sw)
        patches = spatial_patchify_video(x_clip, ph, pw, sh, sw, coords)  # [P,2,pt,ph,pw]
        if patches.shape[0] > N:
            patches = patches[:N]
        B = int(patches.shape[0])
        if B == 0:
            if self.accelerator.is_main_process:
                print("[validate_fixed_t_x0_from_data] no patches after patchify")
            self.model.train()
            return

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        raw_model = self.accelerator.unwrap_model(self.model)

        # one helper to run the net and make x0
        def _predict_x0(x_t: torch.Tensor, t_scalar: float, scaled: bool) -> torch.Tensor:
            t = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
            t_inp = (t * float(self.t_scale)) if scaled else t
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                u = raw_model(x_t.to(torch.bfloat16), t_inp.to(torch.bfloat16)).float()
            return x_t - _append_dims(t, x_t.ndim) * u  # [B,2,pt,ph,pw]

        # Reuse a single noise draw per t
        for t_scalar in t_values:
            # 1) build x_t and the oracle u*
            noise = torch.randn_like(patches)                                   # [B,2,pt,ph,pw]
            t_vec = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
            t_b   = _append_dims(t_vec, patches.ndim)

            x0_true = patches.float()                                           # [B,2,pt,ph,pw]
            x_t     = (1.0 - t_b) * x0_true + t_b * noise                      # [B,2,pt,ph,pw]
            u_star  = (noise - x0_true)                                         # [B,2,pt,ph,pw]
            x0_orac = x_t - t_b * u_star                                        # should equal x0_true

            # 2) run model with (a) scaled t, (b) unscaled t (t-scale sanity check)
            x0_pred_scaled   = _predict_x0(x_t, t_scalar, scaled=True)
            x0_pred_unscaled = _predict_x0(x_t, t_scalar, scaled=False)

            # 3) metrics
            mse_scaled   = torch.mean((x0_pred_scaled   - x0_true).float()**2).item()
            mse_unscaled = torch.mean((x0_pred_unscaled - x0_true).float()**2).item()
            mse_oracle   = torch.mean((x0_orac          - x0_true).float()**2).item()

            if self.accelerator.is_main_process:
                print(f"[fixed-t] t={t_scalar:.3f} | MSE scaled={mse_scaled:.6e}  unscaled={mse_unscaled:.6e}  oracle={mse_oracle:.3e}")
                wandb.log({
                    "val_fixed_t/mse_scaled": mse_scaled,
                    "val_fixed_t/mse_unscaled": mse_unscaled,
                    "val_fixed_t/mse_oracle": mse_oracle,
                    "val_fixed_t/t": float(t_scalar),
                }, step=self.global_step)

            # 4) log a video for the *scaled* prediction (what you care about)
            to_show = x0_pred_scaled  # [B,2,pt,ph,pw]
            rows = int(self.cfg.get("validation", {}).get("grid_rows", 2))
            cols = int(self.cfg.get("validation", {}).get("grid_cols", max(1, (B + rows - 1) // rows)))
            Tvid = int(to_show.shape[2])

            frames = []
            for tt in range(Tvid):
                row_tiles = []
                for r in range(rows):
                    col_tiles = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < B:
                            patch = to_show[idx]  # [2,pt,ph,pw]
                            mag = torch.sqrt(torch.clamp(patch[0, tt]**2 + patch[1, tt]**2, min=0.0))
                        else:
                            mag = torch.zeros_like(to_show[0, 0, tt])
                        col_tiles.append(mag)
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                grid_img = torch.cat(row_tiles, dim=-2)                          # [rows*ph, cols*pw]
                frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

            arr = np.stack(frames, axis=0)                                       # [T,H,W]
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
            if self.accelerator.is_main_process:
                tag = f"val/x0_pred_from_data_t_{t_scalar:.3f}".replace(".", "p")
                wandb.log({tag: vid}, step=self.global_step)

        self.model.train()

    # -------- validation: unconditional pixel patches (no VAE) --------
    @torch.no_grad()
    def validate_uncond(self):
        """
        Unconditional validation in pixel space (run RAW and EMA):
        • sample N independent pixel patches x ~ [2, pt, ph, pw] with Euler(x0)
        • log two grids: val/uncond_patch_grid_pixels_raw, ..._ema
        """
        self.model.eval()
        device = self.accelerator.device
        vcfg = self.cfg.get("validation", {})
        pt = int(vcfg.get("patch_t", 8))
        ph = int(vcfg.get("patch_h", 64))
        pw = int(vcfg.get("patch_w", 64))
        N  = int(vcfg.get("num_uncond_videos", 8))

        scfg   = self.cfg.get("sampler", {})
        steps  = int(scfg.get("num_steps", 18))
        shift  = scfg.get("shift", None)
        sigma_k = float(scfg.get("sigma_exponent", 1.0))

        # flux-style time reparam to build sigmas (1→0)
        seq_len = int(2 * pt * ph * pw)
        if shift is None:
            flux_mu = calculate_flux_mu(
                seq_len,
                x1=float(scfg.get("x1", 256)),  y1=float(scfg.get("y1", 0.5)),
                x2=float(scfg.get("x2", 4096)), y2=float(scfg.get("y2", 1.15)),
                exp_max=float(scfg.get("exp_max", 7.0)),
            )
        else:
            flux_mu = math.log(float(shift))

        sigmas = get_flux_sigmas_from_mu(steps, flux_mu, device=device, dtype=torch.float32)
        if sigma_k != 1.0:
            sigmas = sigmas.pow(sigma_k)

        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

        def _run_variant(which: str):
            # pick weights
            unwrapped = self.accelerator.unwrap_model(self.model)
            if which == "ema":
                self.ema.apply_to(unwrapped)

            noise = torch.randn(N, 2, pt, ph, pw, device=device, dtype=torch.float32)
            x_samples = self._euler_sample_x0(unwrapped, noise, sigmas)  # [N,2,pt,ph,pw]

            # grid (magnitudes)
            rows = int(vcfg.get("grid_rows", 2))
            cols = int(vcfg.get("grid_cols", max(1, (N + rows - 1) // rows)))
            T = int(x_samples.shape[2])
            frames = []
            for tt in range(T):
                row_tiles = []
                for r in range(rows):
                    col_tiles = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < x_samples.shape[0]:
                            patch = x_samples[idx]  # [2,pt,ph,pw]
                            mag = torch.sqrt(torch.clamp(patch[0, tt]**2 + patch[1, tt]**2, min=0.0))
                        else:
                            mag = torch.zeros_like(x_samples[0, 0, tt])
                        col_tiles.append(mag)
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                grid_img = torch.cat(row_tiles, dim=-2)
                frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

            arr = np.stack(frames, axis=0)
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
            if self.accelerator.is_main_process:
                wandb.log({f"val/uncond_patch_grid_pixels_{which}": vid}, step=self.global_step)

            if which == "ema":
                self.ema.restore(unwrapped)

        # run both variants, then restore RAW for training
        _run_variant("ema")
        _run_variant("raw")
        self.model.train()

    @torch.no_grad()
    def _build_zf(self, x_true_2thw: torch.Tensor, R: int):
        """
        Build centered undersampling mask (GRO) + zero-filled complex pixels.
        Returns:
        x_zf_full: [T,H,W] complex
        m_full:    [T,H,W] float (centered ky mask broadcast over kx)
        """
        device = x_true_2thw.device
        _, T, H, W = x_true_2thw.shape
        deg = MRIDeg(pe=H, fr=T, R=R, dsp=0, verbose=False)
        m_TH = torch.from_numpy(deg.mask_ky_t.T.copy()).to(device=device, dtype=torch.float32)  # [T,H]
        m_THW = m_TH[:, :, None].expand(T, H, W)  # [T,H,W]

        # GT → centered k-space → masked → IFFT (zero-filled)
        xr, xi = x_true_2thw[0], x_true_2thw[1]             # [T,H,W]
        xc_true = torch.complex(xr, xi)
        k_true  = torch.fft.fft2(xc_true, norm="ortho")
        k_true_c = torch.fft.fftshift(k_true, dim=(-2, -1))
        k_meas_c = k_true_c * m_THW
        x_zf_full = torch.fft.ifft2(torch.fft.ifftshift(k_meas_c, dim=(-2, -1)), norm="ortho")  # [T,H,W] complex
        return x_zf_full, m_THW

    def _spatial_tiling(self, H, W, ph, pw, overlap_pct):
        sh = pct_to_stride_len(ph, overlap_pct)
        sw = pct_to_stride_len(pw, overlap_pct)
        coords = spatial_coords(H, W, ph, pw, sh, sw)
        return coords, sh, sw

    def _append_dims(self, v: torch.Tensor, target_ndim: int) -> torch.Tensor:
        return v[(...,) + (None,) * (target_ndim - v.ndim)]

    @torch.no_grad()
    def _euler_sample_x0(self, net, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Plain Euler(x0) solver for Flow Matching velocities.
        x:      [N,2,pt,ph,pw]  (float32)
        sigmas: [S+1] from 1.0 → 0.0 (inclusive)
        Returns x in same shape.
        """
        device = x.device
        B = x.shape[0]
        total = sigmas.numel() - 1
        for i in tqdm(range(total)):
            t = sigmas[i].expand(B)
            s = sigmas[i + 1].expand(B)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                u = net(x.to(torch.bfloat16), (t * float(self.t_scale)).to(torch.bfloat16)).float()
            x0_pred = x - self._append_dims(t, x.ndim) * u
            ratio   = self._append_dims((s / t.clamp_min(1e-8)), x.ndim)
            x = ratio * x + (1.0 - ratio) * x0_pred
        return x

    def _temporal_starts(self, T: int, pt: int):
        if pt >= T: return [0]
        step = max(1, pt - 1)  # 1-frame overlap
        starts = list(range(0, T - pt + 1, step))
        if starts[-1] != T - pt:
            starts.append(T - pt)
        return starts

    def _dc_kspace_grad_step(
        self,
        x0_patches: torch.Tensor,        # [P,2,pt,ph,pw], requires_grad=True
        coords, H, W, ph, pw, sh, sw,
        x_zf_chunk_c: torch.Tensor,      # [Tc,H,W] complex (ZF from GT)
        m_THW_chunk: torch.Tensor,       # [Tc,H,W] float mask (centered ky, broadcast over kx)
        step_sz: float,                  # scalar step size (e.g. 1e-1 ~ 10.)
    ) -> torch.Tensor:
        """
        k-space consistency:
        depatchify(x0) -> complex -> FFT2 (centered) -> mask -> IFFT2 -> image
        loss = || image_masked(x0) - ZF(x_true) ||^2_2      (image-space)
        x0 <- x0 - η * ∇_x0 loss
        """
        # 1) depatchify (keeps autograd graph)
        x_full = depatchify2d_over_time(x0_patches, H, W, ph, pw, sh, sw, coords)  # [2,pt,H,W]
        xr, xi = x_full[0], x_full[1]                                              # [pt,H,W]
        x_pred_c = torch.complex(xr, xi)                                           # [pt,H,W]

        Tc = int(x_zf_chunk_c.shape[0])
        x_pred_c = x_pred_c[:Tc]                                                   # [Tc,H,W]

        # 2) FFT2 centered
        k = torch.fft.fft2(x_pred_c, norm="ortho")                                 # [Tc,H,W] complex
        k_c = torch.fft.fftshift(k, dim=(-2, -1))                                  # centered

        # 3) apply centered ky mask (broadcast over kx)
        k_meas_c = k_c * m_THW_chunk.to(k_c.dtype)                                 # [Tc,H,W] complex

        # 4) back to image-space
        x_meas = torch.fft.ifft2(torch.fft.ifftshift(k_meas_c, dim=(-2, -1)), norm="ortho")  # [Tc,H,W] complex

        # 5) image-space L2 against ZF reference
        diff = x_meas - x_zf_chunk_c
        L = (diff.real.pow(2) + diff.imag.pow(2)).mean()

        # 6) gradient wrt patches & update
        (g,) = torch.autograd.grad(L, x0_patches, retain_graph=False, create_graph=False)
        return x0_patches - float(step_sz) * g

    def _inverse_pass_once(self, which: str, x_true: torch.Tensor):
        """
        One full-video inverse solve for weights 'raw' or 'ema'
        """
        device = x_true.device
        vcfg   = self.cfg.get("validation", {})
        ph = int(vcfg.get("patch_h", 64)); pw = int(vcfg.get("patch_w", 64))
        pt = int(vcfg.get("patch_t", 8))
        overlap_pct = float(vcfg.get("overlap_spatial_pct", 5.0))
        R = int(self.cfg.get("deg", {}).get("R", 8))
        dc_lambda = float(self.cfg.get("validation", {}).get("dc_lambda", 0.3))
        dc_every  = max(1, int(self.cfg.get("validation", {}).get("dc_every", 4)))

        # choose weights
        unwrapped = self.accelerator.unwrap_model(self.model)
        if which == "ema":
            self.ema.apply_to(unwrapped)

        _, T, H, W = x_true.shape
        coords, sh, sw = self._spatial_tiling(H, W, ph, pw, overlap_pct)
        P = len(coords)

        print(f'PATCHES: {P}')

        # schedule
        steps = int(self.cfg.get("sampler", {}).get("num_steps", 25))
        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

        # zero-filled reference (complex)
        x_zf_full_c, m_THW_full = self._build_zf(x_true, R=R)  # [T,H,W] complex

        # solve per temporal chunk
        starts = self._temporal_starts(T, pt)
        for t0 in starts:
            t1 = min(t0 + pt, T)
            Tc_valid = int(t1 - t0)
            x_zf_chunk_c = x_zf_full_c[t0:t1]             # [Tc_valid,H,W]
            m_THW_chunk  = m_THW_full[t0:t1]              # [Tc_valid,H,W]
            if Tc_valid < pt:
                pad = pt - Tc_valid
                zero_pad_c = torch.zeros((pad, H, W), device=device, dtype=x_zf_chunk_c.dtype)
                zero_pad_m = torch.zeros((pad, H, W), device=device, dtype=m_THW_chunk.dtype)
                x_zf_chunk_c = torch.cat([x_zf_chunk_c, zero_pad_c], dim=0)   # [pt,H,W]
                m_THW_chunk  = torch.cat([m_THW_chunk,  zero_pad_m], dim=0)   # [pt,H,W]

            # init pixel patches noise
            noise_full = torch.randn((2, pt, H, W), device=device, dtype=torch.float32)
            x = spatial_patchify_video(noise_full, ph, pw, sh, sw, coords).float()  # [P,2,pt,ph,pw]
            B = x.shape[0]; total = sigmas.numel() - 1

            pbar = tqdm(total=total, desc=f"Euler(x0)+pixDC[{which}]", dynamic_ncols=True, leave=False,
                        disable=not self.accelerator.is_main_process)
            for i in tqdm(range(total)):
                t = sigmas[i].expand(B); s = sigmas[i + 1].expand(B)

                # velocity
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        u = unwrapped(x.to(torch.bfloat16), (t * float(self.t_scale)).to(torch.bfloat16)).float()

                # predictor
                x0_pred = x - self._append_dims(t, x.ndim) * u  # [P,2,pt,ph,pw]
                x1_pred = x + self._append_dims(1 - t, x.ndim) * u

                # -- periodic pixel-space DC with grads wrt patches --
                x0_pred = x0_pred.detach().requires_grad_(True)
                step_sz = 15       # [P]
                x0_dc = self._dc_kspace_grad_step(
                    x0_pred, coords, H, W, ph, pw, sh, sw,
                    x_zf_chunk_c[:Tc_valid], m_THW_chunk[:Tc_valid],
                    step_sz=self.cfg.get("validation", {}).get("dc_step_size", 5.0),
                ).detach()
                x0_pred = self._append_dims(t, x.ndim) * x0_dc + self._append_dims(1 - t, x.ndim) * x0_pred
                x1_pred = self._append_dims(s, x.ndim).sqrt() * x1_pred + self._append_dims(1 - s, x.ndim).sqrt() * torch.randn_like(x1_pred)

                x = self._append_dims(1 - s, x.ndim) * x0_pred + self._append_dims(s, x.ndim) * x1_pred

                if self.accelerator.is_main_process:
                    pbar.update(1)
            pbar.close()

            # quick log this chunk (valid frames only)
            if self.accelerator.is_main_process:
                x_full = depatchify2d_over_time(x, H, W, ph, pw, sh, sw, coords)  # [2,pt,H,W]
                frames = []
                for tt in range(Tc_valid):
                    mag = torch.sqrt(torch.clamp(x_full[0, tt]**2 + x_full[1, tt]**2, min=0.0)).unsqueeze(0)
                    frames.append(_frame_to_uint8(mag))
                arr = np.stack(frames, axis=0)
                arr = arr[:, None, :, :]
                arr = np.repeat(arr, 3, axis=1)
                vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
                wandb.log({f"val/inverse_dc_chunk_{which}_{t0:04d}": vid}, step=self.global_step)

            break

        if which == "ema":
            self.ema.restore(unwrapped)

    def validate_inverse_dc(self):
        """
        Run inverse validation for both weight sets (RAW + EMA).
        """
        self.model.eval()
        device = self.accelerator.device
        # fetch one val sample deterministically
        try:
            val_it = iter(self.val_dl)
            batch_list = next(val_it)
            x_true = None
            for item in batch_list:
                x_true = item
                break
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[validate_inverse_dc] empty val loader")
            return

        x_true = x_true.to(device=device, dtype=torch.float32)
        if x_true.dim() == 3:
            x_true = x_true.unsqueeze(1)
        # run both variants
        self._inverse_pass_once("ema", x_true)
        self._inverse_pass_once("raw", x_true)
        self.model.train()

    # -------- checkpoint --------
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
            "ema": self.ema.shadow,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(save_dir, "state.pt")
        self.accelerator.save(state, path)
        if self.accelerator.is_main_process:
            wandb.save(path)
            self.accelerator.print(f"Saved checkpoint: {save_dir}")


# ==================== DataLoader builder ====================

def build_dataloader(ds_cfg: Dict[str, Any], dl_cfg: Dict[str, Any], is_train: bool) -> DataLoader:
    DS = try_import_dataset(ds_cfg["name"])
    dataset = DS(**ds_cfg.get("args", {}))  # CINEDataset (pixels)
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


# ==================== main ====================

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(42)

    train_dl = build_dataloader(cfg["train_dataset"], cfg["dataloader"], is_train=True)
    val_dl   = build_dataloader(cfg["val_dataset"],   cfg["dataloader"], is_train=False)

    ModelClass = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = ModelClass(**cfg["model"]["args"]).to(torch.bfloat16)

    # Optional pretrained / resume-aware load
    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))
    resume_flag = bool(cfg["model"].get("resume", False))

    print(resume_flag)

    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if (not resume_flag) and ("ema" in ckpt):
            print(f"[FM] loaded EMA weights from {pretrained_path}")
            state = ckpt["ema"]
        else:
            print(f"[FM] loaded non-EMA weights from {pretrained_path}")
            state = ckpt.get("model", ckpt)
        new_sd = OrderedDict((k[10:] if k.startswith("_orig_mod.") else k, v) for k, v in state.items())
        missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
        if not strict_load:
            print(f"[FM] missing={len(missing)} unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FM] total number of model parameters: {n_params/1e9:.3f}B")

    trainer = PixelFMTrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/flow_matching_pixel.yaml")
    args = ap.parse_args()
    main(args.config)
