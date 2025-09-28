#!/usr/bin/env python3
import os, sys, time, importlib, math, yaml
from typing import Dict, Any, List
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

from utils.ema import Ema
from data.deg import MRIDeg  # for the inverse/DC validator

# ==================== tiny helpers ====================

def ragged_collate(batch):
    # our dataset already returns fixed (2,8,64,64); stacking is safe, but we keep this for compatibility
    return batch

def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def try_import_dataset(name: str):
    if name == "CINEPixelDataset":
        mod = importlib.import_module("data.cine_dataset_pixel")
        return getattr(mod, "CINEPixelDataset")
    raise ValueError(f"Unknown dataset '{name}'.")

def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

# ==================== 2D spatial patchify (val) ====================

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

# ==================== FFT helpers ====================

def fft2c(x_2thw: torch.Tensor) -> torch.Tensor:
    xr, xi = x_2thw[0], x_2thw[1]
    xc = torch.complex(xr, xi)
    k  = torch.fft.fft2(xc, norm="ortho")
    kc = torch.fft.fftshift(k, dim=(-2, -1))
    return torch.stack((kc.real, kc.imag), dim=0)

def ifft2c(kc_2thw: torch.Tensor) -> torch.Tensor:
    kr, ki = kc_2thw[0], kc_2thw[1]
    kc = torch.complex(kr, ki)
    k  = torch.fft.ifftshift(kc, dim=(-2, -1))
    x  = torch.fft.ifft2(k, norm="ortho")
    return torch.stack((x.real, x.imag), dim=0)

# ==================== Trainer ====================

class PixelFMTrainer:
    """
    Flow-Matching directly in pixel space on fixed patches [2,8,64,64] (bf16 forward).
    """
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        cfg['logging']['out_dir'] = os.path.join(cfg['logging']['out_dir'], "flowmatch_pixels")
        self.cfg = cfg
        self.model = torch.compile(model, fullgraph=False)

        # keep t_scale at 1000 unless overridden
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

        # -------- Validation patch config (defaults to 8×64×64) --------
        vcfg = self.cfg.get("validation", {})
        self.val_patch_h  = int(vcfg.get("patch_h", 64))
        self.val_patch_w  = int(vcfg.get("patch_w", 64))
        self.val_patch_t  = int(vcfg.get("patch_t", 8))
        self.val_patch_bs = int(vcfg.get("patch_batch", 64))

    # -------- Rectified-Flow loss on PIXELS --------
    def _rectified_flow_loss_pixels(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        X: [B,2,8,64,64] (bf16)
        x_t=(1-t)X + t N, target u*=N - X
        """
        device = self.accelerator.device
        B = X.shape[0]
        noise  = torch.randn_like(X)                                    # bf16
        sigmas = torch.rand((B,), device=device, dtype=torch.bfloat16)  # U(0,1)

        t_b    = sigmas.view(B, *([1]*(X.ndim-1)))
        x_t    = (1.0 - t_b) * X + t_b * noise
        target = (noise - X)

        t_inp  = (sigmas * self.t_scale)                                # keep t_scale=1000
        pred = self.model(x_t, t_inp)                                   # bf16 velocity

        if self.accelerator.is_main_process:
            print(f'[RF] t_scale={self.t_scale}  data[min,max]=({X.min():.4f},{X.max():.4f}) '
                  f'pred_norm={pred[0].norm():.4f} target_norm={target[0].norm():.4f} t={sigmas[0]:.4f}')

        mse  = torch.nn.functional.mse_loss(pred.float(), target.float())
        return {"total": mse, "mse": mse}

    def _fm_loss(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._rectified_flow_loss_pixels(X)

    # -------- training step --------
    def compute_loss(self, batch_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # our dataset yields identical shapes; safe to stack
        X = torch.stack(batch_list, dim=0)  # [B,2,8,64,64]
        return self._fm_loss(X)

    # -------- training loop --------
    def train(self):
        log_cfg = self.cfg["logging"]
        self.accelerator.print("Starting Pixel Flow Matching training (bf16)…")

        pbar = tqdm(
            total=self.total_steps,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            desc="train",
            leave=True,
        )
        last = time.perf_counter()
        train_iter = iter(self.train_dl)

        for step in range(self.total_steps):
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
                        print(f"[warn] non-finite loss at step {step}; masking to 0.")
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

                    if self.accelerator.is_main_process and (step % log_cfg["log_every_steps"] == 0):
                        scalars = {f"train/{k}": float(v.detach().cpu()) for k, v in losses.items()}
                        scalars["lr"] = self.optimizer.param_groups[0]["lr"]
                        wandb.log(scalars, step=step)

                    if (step % log_cfg["val_every_steps"] == 0) and step > 0:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[val] step {step}")
                        self.validate_fixed_t_x0_from_data((0.01, 0.10, 0.25, 0.50, 0.75, 1.00))
                        self.validate_uncond()
                        self.accelerator.wait_for_everyone()

                    if (step % log_cfg["save_every_steps"] == 0) and step > 0:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[ckpt] step {step}")
                        self.save_checkpoint(step)
                        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            pbar.close()
            self.accelerator.print("Training complete.")

    @torch.no_grad()
    def validate_fixed_t_x0_from_data(self, t_values=(0.01, 0.10, 0.25, 0.50, 0.75, 0.999)):
        """
        Pull one val sample [2,T,H,W], center-clip to pt=8, spatial patchify to 64×64 (no overlap),
        build x_t, predict u, compute x0_pred, log MSEs and a grid video.
        """
        self.model.eval()
        device = self.accelerator.device

        pt = int(self.val_patch_t); ph = int(self.val_patch_h); pw = int(self.val_patch_w)
        N  = int(self.cfg.get("validation", {}).get("num_uncond_videos", 8))

        # fetch one val sample
        try:
            val_it = iter(self.val_dl)
            batch_list = next(val_it)
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[validate_fixed_t_x0_from_data] empty val loader")
            self.model.train()
            return

        x_true = batch_list[0].to(device=device, dtype=torch.float32)  # [2,8,64,64]
        _, T, H, W = x_true.shape

        # if val examples already 8×64×64, this is a no-op
        if T >= pt:
            t0 = (T - pt) // 2
            x_clip = x_true[:, t0:t0 + pt]
        else:
            reps = (pt + T - 1) // T
            x_clip = x_true.repeat(1, reps, 1, 1)[:, :pt]

        sh = ph; sw = pw
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

        def _predict_x0(x_t: torch.Tensor, t_scalar: float, scaled: bool) -> torch.Tensor:
            t = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
            t_inp = (t * float(self.t_scale)) if scaled else t
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                u = raw_model(x_t.to(torch.bfloat16), t_inp.to(torch.bfloat16)).float()
            return x_t - _append_dims(t, x_t.ndim) * u

        for t_scalar in t_values:
            noise = torch.randn_like(patches)
            t_vec = torch.full((B,), float(t_scalar), device=device, dtype=torch.float32)
            t_b   = _append_dims(t_vec, patches.ndim)

            x0_true = patches.float()
            x_t     = (1.0 - t_b) * x0_true + t_b * noise
            u_star  = (noise - x0_true)
            x0_orac = x_t - t_b * u_star

            x0_pred_scaled   = _predict_x0(x_t, t_scalar, scaled=True)
            x0_pred_unscaled = _predict_x0(x_t, t_scalar, scaled=False)

            mse_scaled   = torch.mean((x0_pred_scaled   - x0_true).float()**2).item()
            mse_unscaled = torch.mean((x0_pred_unscaled - x0_true).float()**2).item()
            mse_oracle   = torch.mean((x0_orac          - x0_true).float()**2).item()

            if self.accelerator.is_main_process:
                print(f"[fixed-t] t={t_scalar:.3f} | MSE scaled(×{self.t_scale:.0f})={mse_scaled:.6e}  "
                      f"unscaled={mse_unscaled:.6e}  oracle={mse_oracle:.3e}")
                wandb.log({
                    "val_fixed_t/mse_scaled": mse_scaled,
                    "val_fixed_t/mse_unscaled": mse_unscaled,
                    "val_fixed_t/mse_oracle": mse_oracle,
                    "val_fixed_t/t": float(t_scalar),
                })

            # log grid for the scaled case only
            to_show = x0_pred_scaled
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
                            patch = to_show[idx]
                            mag = torch.sqrt(torch.clamp(patch[0, tt]**2 + patch[1, tt]**2, min=0.0))
                        else:
                            mag = torch.zeros_like(to_show[0, 0, tt])
                        col_tiles.append(mag)
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                grid_img = torch.cat(row_tiles, dim=-2)
                frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

            arr = np.stack(frames, axis=0)
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
            if self.accelerator.is_main_process:
                tag = f"val/x0_pred_from_data_t_{t_scalar:.3f}".replace(".", "p")
                wandb.log({tag: vid})

        self.model.train()

    # -------- unconditional patch sampling (Euler x0) --------
    @torch.no_grad()
    def validate_uncond(self):
        self.model.eval()
        device = self.accelerator.device
        vcfg = self.cfg.get("validation", {})
        pt = int(vcfg.get("patch_t", 8))
        ph = int(vcfg.get("patch_h", 64))
        pw = int(vcfg.get("patch_w", 64))
        N  = int(vcfg.get("num_uncond_videos", 8))
        steps  = int(self.cfg.get("sampler", {}).get("num_steps", 18))

        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        def _run_variant(which: str):
            unwrapped = self.accelerator.unwrap_model(self.model)
            if which == "ema":
                self.ema.apply_to(unwrapped)

            x = torch.randn(N, 2, pt, ph, pw, device=device, dtype=torch.float32)
            total = sigmas.numel() - 1
            for i in range(total):
                t = sigmas[i].expand(N); s = sigmas[i + 1].expand(N)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    u = unwrapped(x.to(torch.bfloat16), (t * float(self.t_scale)).to(torch.bfloat16)).float()
                x0_pred = x - _append_dims(t, x.ndim) * u
                ratio   = _append_dims((s / t.clamp_min(1e-8)), x.ndim)
                x = ratio * x + (1.0 - ratio) * x0_pred

            # grid (magnitudes)
            rows = int(vcfg.get("grid_rows", 2))
            cols = int(vcfg.get("grid_cols", max(1, (N + rows - 1) // rows)))
            T = int(x.shape[2])
            frames = []
            for tt in range(T):
                row_tiles = []
                for r in range(rows):
                    col_tiles = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < x.shape[0]:
                            patch = x[idx]
                            mag = torch.sqrt(torch.clamp(patch[0, tt]**2 + patch[1, tt]**2, min=0.0))
                        else:
                            mag = torch.zeros_like(x[0, 0, tt])
                        col_tiles.append(mag)
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                grid_img = torch.cat(row_tiles, dim=-2)
                frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

            arr = np.stack(frames, axis=0)
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
            if self.accelerator.is_main_process:
                wandb.log({f"val/uncond_patch_grid_pixels_{which}": vid})

            if which == "ema":
                self.ema.restore(unwrapped)

        _run_variant("ema")
        _run_variant("raw")
        self.model.train()

    # -------- checkpoint --------
    def save_checkpoint(self, step: int):
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        save_dir = os.path.join(self.cfg["logging"]["out_dir"], f"step_{step:07d}")
        os.makedirs(save_dir, exist_ok=True)

        state = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "global_step": step,
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
    dataset = DS(**ds_cfg.get("args", {}))  # CINEPixelDataset expects: data_path, t_frames, crop_hw, apply_preproc, ...
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
