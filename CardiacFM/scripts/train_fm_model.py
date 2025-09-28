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
import gc

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# EMA
from utils.ema import Ema

# UniPC Flow Matching sampler (expects model(x, t, **extra_args))
from CardiacFM.sampler.flow_match_uni_pc import sample_unipc
from data.deg import MRIDeg

# model = LatentFlowMatchTransformer(...).cuda().eval()
# # (optional) load fitted coeffs
# import json
# with open('teacache_poly.json', 'r') as f:
#     coeffs = json.load(f)['coeffs_high_to_low']

# # initialize for a T-step trajectory (e.g., UniPC/DPMSolver schedule)
# T = 25
# model.initialize_teacache(enable_teacache=True, num_steps=T, rel_l1_thresh=0.15, poly_coeffs=coeffs)

# # during inference (no extra code needed); the model caches internally across calls
# for i, t in enumerate(t_schedule):
#     t_batch = torch.full((B,), float(t), device=latents.device)
#     vel = model(latents, t_batch)
#     # ... sampler update ...


# ==================== FramePack-style time utilities ====================

def flux_time_shift(t: torch.Tensor, mu=1.15, sigma: float = 1.0) -> torch.Tensor:
    """
    t: 1D or ND tensor in [0,1].
    mu: scalar (float or 0-dim tensor). We keep it as a tensor on t's device/dtype.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = torch.clamp(t, min=1e-5, max=1.0)

    mu_t = torch.as_tensor(mu, device=t.device, dtype=torch.float32)  # scalar tensor
    if mu_t.numel() != 1:
        raise ValueError(f"mu must be scalar; got shape {tuple(mu_t.shape)}")
    emu = torch.exp(mu_t)  # exp(mu) in fp32

    return emu / (emu + (1.0 / t - 1.0).pow(sigma))


def get_flux_sigmas_from_mu(n_steps: int, mu, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return (n_steps+1) schedule 1→0 (inclusive), reparam via flux_time_shift.
    Keep the computation in fp32 for stability; cast at the end.
    """
    # build t in fp32 for numerics, then cast to requested dtype at return
    t_f32 = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device, dtype=torch.float32)
    sig = flux_time_shift(t_f32, mu=mu)              # fp32
    return sig.to(dtype=dtype)


def calculate_flux_mu(context_length: int,
                      x1: float = 256, y1: float = 0.5,
                      x2: float = 4096, y2: float = 1.15,
                      exp_max: float = 7.0) -> float:
    """
    Linear map from context length → mu, capped so exp(mu) ≤ exp_max.
    """
    k = (y2 - y1) / max(1.0, (x2 - x1))
    b = y1 - k * x1
    mu = k * float(context_length) + b
    return float(min(mu, math.log(exp_max)))


# ==================== ragged collate + dynamic imports ====================

def ragged_collate(batch):
    return batch

def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def try_import_dataset(name: str):
    """
    TRAIN: CINEFlowMatchLatentDataset (precomputed latents)
    VAL/TEST: CINEDataset (raw) – unchanged
    """
    if name == "CINEFlowMatchLatentDataset":
        mod = importlib.import_module("data.cine_flow_dataset_new")
        return getattr(mod, "CINEFlowMatchLatentDataset")
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

def _to_wandb_video_one(x_mag_1t: torch.Tensor, fps: int = 7):
    # x_mag_1t: [1, T, H, W]
    T = int(x_mag_1t.shape[1])
    frames = [_frame_to_uint8(x_mag_1t[:, t]) for t in range(T)]
    arr = np.stack(frames, axis=0)  # [T, H, W]
    arr = arr[:, None, :, :]
    arr = np.repeat(arr, 3, axis=1)
    return wandb.Video(arr, fps=int(fps), format="mp4")


# ==================== 2D patchify (val) with overlap-add ====================

def temporal_starts_full_coverage(T_total: int, T_win: int, prefer_stride: int) -> List[int]:
    if T_total <= T_win: return [0]
    s = max(1, int(prefer_stride))  # normally T-1 → 1-frame overlap
    starts = list(range(0, T_total - T_win + 1, s))
    if starts[-1] + T_win < T_total:
        n = len(starts) + 1
        s = max(1, math.floor((T_total - T_win) / (n - 1)))
        starts = [i * s for i in range(n - 1)]
        last = T_total - T_win
        if starts[-1] != last:
            starts.append(last)
    return starts

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

@torch.no_grad()
def depatchify2d_fixed_stride(patches: torch.Tensor, out_hw: Tuple[int,int], ph: int, pw: int, sh: int, sw: int) -> torch.Tensor:
    H, W = out_hw
    n1 = max(1, math.ceil((H - ph) / sh) + 1)
    n2 = max(1, math.ceil((W - pw) / sw) + 1)
    expected = n1 * n2
    if int(patches.shape[0]) != expected:
        raise RuntimeError(f"N={patches.shape[0]} but expect {expected} for H,W={out_hw}, ph,pw={ph,pw}, sh,sw={sh,sw}")
    out_num = torch.zeros((H, W), dtype=patches.dtype, device=patches.device)
    out_den = torch.zeros((H, W), dtype=torch.float32, device=patches.device)

    def axis_weights(L_eff: int, idx: int, n: int, O: int) -> torch.Tensor:
        has_prev = (idx > 0)
        has_next = (idx < n - 1)
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
        w = torch.ones(L_eff, dtype=torch.float32, device=patches.device)
        if L_left > 0:
            w[:L_left] = 0.5 if L_left == 1 else torch.linspace(0.0, 1.0, steps=L_left, device=patches.device)
        if L_right > 0:
            w[-L_right:] = 0.5 if L_right == 1 else torch.linspace(1.0, 0.0, steps=L_right, device=patches.device)
        return w

    O1 = max(0, ph - sh); O2 = max(0, pw - sw)
    idx = 0
    for j in range(n1):
        y0 = j * sh; y1 = min(y0 + ph, H); s1 = y1 - y0; w1 = axis_weights(s1, j, n1, O1)
        for k in range(n2):
            x0 = k * sw; x1 = min(x0 + pw, W); s2 = x1 - x0; w2 = axis_weights(s2, k, n2, O2)
            w = (w1[:, None] * w2[None, :])
            p = patches[idx][:s1, :s2]
            out_num[y0:y1, x0:x1] += (p * w).to(out_num.dtype)
            out_den[y0:y1, x0:x1] += w
            idx += 1
    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)


# ==================== Trainer ====================

class LatentFMTrainer:
    """
    Train on precomputed latent clips (z) only (bf16).
    TRAIN sample (dataset): list of [Cz,7,P,H',W'] tensors → flatten P to batch.
    VAL: unconditional sample → decode with VAE → log visuals (no GT).
    """
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        cfg['logging']['out_dir'] = os.path.join(cfg['logging']['out_dir'], "flowmatch")
        self.cfg = cfg
        self.model = model
        self.model = torch.compile(self.model, fullgraph=False)

        self.t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))  # <<< consistent t scaling

        proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)
        # Use bf16 mixed precision
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
                # optimizer
                if "optimizer" in resume_state:
                    self.optimizer.load_state_dict(resume_state["optimizer"])
                # scheduler
                if self.scheduler is not None and "scheduler" in resume_state:
                    self.scheduler.load_state_dict(resume_state["scheduler"])
                # step
                self.global_step = int(resume_state.get("global_step", 0))
                print(f"[resume] loaded opt/sched and step={self.global_step} from {resume_path}")
            except Exception as e:
                print(f"[resume] failed to load optimizer/scheduler from {resume_path}: {e}")

        self.start_step = self.global_step  # for tqdm remainder

        # prepare under accelerate
        if self.scheduler is not None:
            (self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler)
        else:
            (self.model, self.optimizer, self.train_dl, self.val_dl) = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema = Ema(unwrapped, decay=float(self.cfg["optim"].get("ema_decay", 0.999)))
        if resume_state is not None and "ema" in resume_state:
            ema_state = {k: v.to(self.accelerator.device) for k, v in resume_state["ema"].items()}
            try:
                self.ema.load_shadow(ema_state)
                print("[resume] restored EMA shadow")
            except Exception as e:
                print(f"[resume] failed to restore EMA: {e}")
                
        self.grad_clip = float(opt_cfg.get("grad_clip", 0.0))

        # wandb dirs
        wdir  = self.cfg["logging"].get("wandb_dir", None)
        wcache = self.cfg["logging"].get("wandb_cache_dir", None)
        if wdir:   os.environ["WANDB_DIR"] = str(wdir)
        if wcache: os.environ["WANDB_CACHE_DIR"] = str(wcache)

        if self.accelerator.is_main_process:
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"].get("run_name", "latent_fm_bf16"),
                config=cfg,
                dir=wdir,
            )

        # -------- Validation patch config --------
        vcfg = self.cfg.get("validation", {})
        self.val_patch_h  = int(vcfg.get("patch_h", 80))
        self.val_patch_w  = int(vcfg.get("patch_w", 80))
        self.val_patch_t  = int(vcfg.get("patch_t", 7))
        self.val_patch_bs = int(vcfg.get("patch_batch", 64))

        # ---------- VAE for validation decode ----------
        vae_cfg = self.cfg.get("vae", {})
        VAE = dynamic_import(vae_cfg["import_path"], vae_cfg["class_name"])
        self.vae = VAE(**vae_cfg.get("args", {}))
        vae_ckpt = self.cfg.get("model", {}).get("vae_ckpt", None) or vae_cfg.get("load_state_dict_from", None)
        if vae_ckpt and os.path.isfile(vae_ckpt):
            ckpt = torch.load(vae_ckpt, map_location="cpu")
            state = ckpt.get("ema", ckpt.get("model", ckpt))
            state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
            self.vae.load_state_dict(state, strict=False)
            if self.accelerator.is_main_process:
                print(f"[VAE] loaded validation decoder weights from {vae_ckpt}")
        self.vae = self.vae.to(self.accelerator.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    # -------- Rectified-Flow loss (bf16 forward, fp32 loss reduce) --------
    def _rectified_flow_loss(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Z: [N, Cz, 7, H', W'] (bf16)
        x0=Z, x1~N(0,1), t~U(0,1)
        x_t=(1-t)Z + t x1, target u*=x1 - Z
        """
        device = Z.device
        N = Z.shape[0]

        # bf16 noise and t
        noise  = torch.randn_like(Z)  # same dtype as Z (bf16 under mixed-precision)
        sigmas = torch.rand((N,), device=device, dtype=torch.bfloat16)

        # broadcast t and mix states
        t_b = sigmas.view(N, *([1] * (Z.dim() - 1)))                # bf16
        x_t = (1.0 - t_b) * Z + t_b * noise                          # bf16
        target = (noise - Z)                                         # bf16

        # time input scaled by 1000.0 (keep float, no .long())
        t_inp = (sigmas * self.t_scale)                              # bf16

        # model forward in bf16 (Accelerate autocast)
        pred = self.model(x_t, t_inp)                                # bf16

        # loss in fp32 for stability
        mse = torch.nn.functional.mse_loss(pred.float(), target.float())
        return {"total": mse, "mse": mse}

    def _fm_loss(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._rectified_flow_loss(Z)

    # -------- training step (latents only) --------
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        """
        TRAIN batch: tensor [B, Cz, nt, H', W'] (bf16 under Accelerate autocast)
        """
        device = self.accelerator.device
        if isinstance(batch, list):
            # If a custom dataloader slips a list in, stack it defensively.
            batch = torch.stack(batch, dim=0)
        Z = batch.to(device=device, dtype=torch.bfloat16)   # [B,Cz,nt,H',W']

        # Rectified flow loss (same math as before, just no permutation/flattening)
        N = Z.shape[0]
        noise  = torch.randn_like(Z)                        # bf16
        if self.global_step < 100000:
            t_vec  = torch.rand((N,), device=device, dtype=torch.bfloat16).sqrt()   # U(0,1)
        else:
            t_vec  = torch.rand((N,), device=device, dtype=torch.bfloat16)   # U(0,1)

        t_b = t_vec.view(N, *([1] * (Z.dim() - 1)))        # [B,1,1,1,1]
        x_t = (1.0 - t_b) * Z + t_b * noise                # bf16
        target = (noise - Z)                                # bf16
        t_inp = (t_vec * self.t_scale)                     # bf16

        pred = self.model(x_t, t_inp)                      # bf16
        mse = torch.nn.functional.mse_loss(pred.float(), target.float())
        return {"total": mse, "mse": mse}

    # -------- training loop --------
    def train(self):
        log_cfg = self.cfg["logging"]
        opt_cfg = self.cfg["optim"]

        self.accelerator.print("Starting Flow Matching training (bf16)…")

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

                    if (self.global_step % log_cfg["val_every_steps"] == 0):# and self.global_step > 0:
                        if self.accelerator.is_main_process:
                            pbar.write(f"[val] step {self.global_step}")
                        # self.validate()
                        torch.cuda.empty_cache()
                        self.validate_fixed_t_x0_from_data()
                        self.validate_fixed_t_x0_from_data(use_ema=True)
                        self.validation_simple()
                        self.validation_simple(use_ema=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.accelerator.wait_for_everyone()
                        self.validation_simple_inv()
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.accelerator.wait_for_everyone()
                        self.validation_simple_inv(use_ema=True)
                        torch.cuda.empty_cache()
                        gc.collect()
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

    # -------- validation: unconditional sample → decode --------
    @torch.no_grad()
    def validate(self):
        """
        Unconditional validation:
          • infer latent (Cz, nt, H', W') via a dummy encode
          • sample N independent latent patches with UniPC (bf16, t scaled ×1000)
          • decode with VAE (fp32) → 8 videos
          • log a 2×4 grid video of magnitudes
        """
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema.apply_to(unwrapped)

        device = self.accelerator.device
        vcfg = self.cfg.get("validation", {})
        pt = int(vcfg.get("patch_t", 7))
        ph = int(vcfg.get("patch_h", 80))
        pw = int(vcfg.get("patch_w", 80))
        bs = int(vcfg.get("patch_batch", 64))
        N  = int(vcfg.get("num_uncond_videos", 8))  # how many independent patch videos

        # ---- latent spatial size via dummy encode (fp32 VAE)
        dummy = torch.zeros(1, 2, pt, ph, pw, device=self.accelerator.device, dtype=torch.float32)
        vae_out = self.vae([dummy], op="encode")
        # Handle (mu, logv) or [mu] returns
        z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
        Cz, nt, Hlat, Wlat = int(z_mu.shape[1]), int(z_mu.shape[2]), int(z_mu.shape[-2]), int(z_mu.shape[-1])

        # ---- sampler schedule (scalar mu for time reparam)
        scfg   = self.cfg.get("sampler", {})
        steps  = int(scfg.get("num_steps", 18))
        shift  = scfg.get("shift", None)  # if set: mu = ln(shift)
        sigma_k = float(scfg.get("sigma_exponent", 1.0))

        seq_len = int(pt * Hlat * Wlat)
        if shift is None:
            flux_mu = calculate_flux_mu(
                seq_len,
                x1=float(scfg.get("x1", 256)),
                y1=float(scfg.get("y1", 0.5)),
                x2=float(scfg.get("x2", 4096)),
                y2=float(scfg.get("y2", 1.15)),
                exp_max=float(scfg.get("exp_max", 7.0)),
            )
        else:
            flux_mu = math.log(float(shift))

        # keep scheduler math in fp32; cast later if you want
        sigmas = get_flux_sigmas_from_mu(steps, flux_mu, device=self.accelerator.device, dtype=torch.float32)
        if sigma_k != 1.0:
            sigmas = sigmas.pow(sigma_k)

        # wrapper to apply t_scale and bf16 inside sampler (unchanged)
        def model_bf16_tscaled(x, t, **kwargs):
            x_bf16 = x.to(torch.bfloat16)
            t_bf16 = (t.to(torch.float32) * self.t_scale).to(torch.bfloat16)
            return self.model(x_bf16, t_bf16, **kwargs)

        # --- add these two small helpers near the top of validate(), after you build `sigmas` ---

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            # expand [B] time vector across spatial/temporal dims of x
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        @torch.no_grad()
        def _model_x0_wrapper(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
            """
            Your model outputs velocity u. UniPC wants x0.
            x0 = x - t * u
            - sampler passes unscaled t in [0,1]
            - model still sees t * self.t_scale (== 1000.0 by default)
            - compute in fp32 for stability; return original dtype
            """
            x_dtype = x.dtype
            x_bf16  = x.to(torch.bfloat16)

            t_unscaled = t.to(torch.float32)                           # [B]
            t_model    = (t_unscaled * self.t_scale).to(torch.bfloat16)

            u = self.model(x_bf16, t_model, **(kwargs or {}))          # predict velocity u
            x0 = x_bf16.float() - _append_dims(t_unscaled, x_bf16.ndim) * u.float()

            return x0.to(x_dtype)


        # ---- unconditional latent sampling: N independent patches ----
        noise = torch.randn(N, Cz, nt, Hlat, Wlat, device=device, dtype=torch.float32)
        print(f"noise shape: {noise.shape}")
        z_samples = sample_unipc(
            _model_x0_wrapper,
            noise,                          # bf16
            sigmas,                         # float32
            extra_args={},                  # explicit empty dict
            callback=None,
            disable=False,                  # show tqdm
            variant=scfg.get("variant", "bh1"),
        )

        # ---- decode in chunks (fp32 for safety) ----
        def _decode_latents(z_bchw_t):
            outs = []
            for i in range(0, z_bchw_t.shape[0], bs):
                chunk = z_bchw_t[i:i+bs].float()
                z_list = [z.unsqueeze(0) for z in chunk]       # list of [1,Cz,nt,H',W']
                dec = self.vae(z_list, op="decode")            # list of [1,2,pt,ph,pw] (pt should match nt's decode)
                # 'dec' may already be tensors; normalize to [2,pt,ph,pw]
                for o in dec:
                    if isinstance(o, (list, tuple)):
                        o = o[0]
                    outs.append(o.squeeze(0))                  # [2,pt,ph,pw]
            return outs

        xhat_list = _decode_latents(z_samples)                 # len=N

        # ---- assemble a 2×4 grid video of magnitudes ----
        rows = int(vcfg.get("grid_rows", 2))
        cols = int(vcfg.get("grid_cols", max(1, (N + rows - 1) // rows)))

        T = int(xhat_list[0].shape[1])  # pt
        frames = []
        for t in range(T):
            row_tiles = []
            for r in range(rows):
                col_tiles = []
                for c in range(cols):
                    idx = r * cols + c
                    if idx < len(xhat_list):
                        patch = xhat_list[idx]                  # [2,pt,ph,pw]
                        mag = torch.sqrt(torch.clamp(patch[0, t]**2 + patch[1, t]**2, min=0.0))
                    else:
                        mag = torch.zeros_like(xhat_list[0][0, t])
                    col_tiles.append(mag)
                row_tiles.append(torch.cat(col_tiles, dim=-1))
            grid_img = torch.cat(row_tiles, dim=-2)             # [rows*ph, cols*pw]
            frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

        arr = np.stack(frames, axis=0)                           # [T, Htot, Wtot]
        arr = arr[:, None, :, :]
        arr = np.repeat(arr, 3, axis=1)
        grid_vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")

        if self.accelerator.is_main_process:
            wandb.log({"val/uncond_patch_grid": grid_vid}, step=self.global_step)

        self.ema.restore(unwrapped)
        self.model.train()

    @torch.no_grad()
    def validate_fixed_t_x0_from_data(self, use_ema: bool = False, t_values=(0.01, 0.10, 0.25, 0.50, 0.75, 1.00)):
        """
        Latent-space version of 'predict x0 from data':
        • take one val video [2,T,H,W]
        • center temporal clip of length pt, spatially patchify (no overlap)
        • VAE-encode patches -> z_patches [B,Cz,nt,H',W']
        • for each t: x_t = (1-t)z + t·ε ; u = net(x_t, t·t_scale); x0 = x_t - t·u
        • decode x0 -> pixel patches; log a grid video of magnitudes per t
        """
        self.model.eval()
        if use_ema:
            self.ema.apply_to(self.model)
        device = self.accelerator.device
        vcfg   = self.cfg.get("validation", {})
        pt     = int(vcfg.get("patch_t", 7))
        ph     = int(vcfg.get("patch_h", 80))
        pw     = int(vcfg.get("patch_w", 80))
        Nshow  = int(vcfg.get("num_uncond_videos", 8))  # how many patches to visualize
        bs     = int(vcfg.get("patch_batch", 64))       # VAE micro-batch

        # -- grab one val sample deterministically
        try:
            it = iter(self.val_dl)
            batch_list = next(it)
            x_true = None
            for item in batch_list:
                x_true = item
                break
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[validate_fixed_t_x0_from_data] empty val loader")
            self.model.train()
            return

        x_true = x_true.to(device=device, dtype=torch.float32)   # [2,T,H,W] or [2,1,H,W]
        if x_true.dim() == 3:
            x_true = x_true.unsqueeze(1)
        _, T, H, W = x_true.shape

        # -- center temporal clip of length pt (wrap if needed)
        if T >= pt:
            t0 = (T - pt) // 2
            x_clip = x_true[:, t0:t0 + pt]                       # [2,pt,H,W]
        else:
            reps = (pt + T - 1) // T
            x_clip = x_true.repeat(1, reps, 1, 1)[:, :pt]        # [2,pt,H,W]

        # -- spatial patchify (no overlap): stride = patch
        sh, sw = ph, pw
        def _spatial_coords(H: int, W: int, ph: int, pw: int, sh: int, sw: int):
            n1 = max(1, math.ceil((H - ph) / sh) + 1)
            n2 = max(1, math.ceil((W - pw) / sw) + 1)
            coords = []
            for j in range(n1):
                y0 = j * sh; y1 = min(y0 + ph, H); s1 = y1 - y0
                for k in range(n2):
                    x0 = k * sw; x1 = min(x0 + pw, W); s2 = x1 - x0
                    coords.append((y0, y1, s1, x0, x1, s2))
            return coords, n1, n2

        def _spatial_patchify_video(x_2thw: torch.Tensor, ph: int, pw: int, sh: int, sw: int, coords):
            _, Tt, Hh, Ww = x_2thw.shape
            out = torch.zeros((len(coords), 2, Tt, ph, pw), dtype=x_2thw.dtype, device=x_2thw.device)
            for idx, (y0,y1,s1,x0,x1,s2) in enumerate(coords):
                patch = torch.zeros((2, Tt, ph, pw), dtype=x_2thw.dtype, device=x_2thw.device)
                patch[:, :, :s1, :s2] = x_2thw[:, :, y0:y1, x0:x1]
                out[idx] = patch
            return out

        coords, n1, n2 = _spatial_coords(H, W, ph, pw, sh, sw)
        patches_px = _spatial_patchify_video(x_clip, ph, pw, sh, sw, coords)   # [P,2,pt,ph,pw]
        if patches_px.shape[0] > Nshow:
            mid = patches_px.shape[0] // 2
            patches_px = patches_px[mid - Nshow//2:mid + Nshow//2]
        B = int(patches_px.shape[0])
        if B == 0:
            if self.accelerator.is_main_process:
                print("[validate_fixed_t_x0_from_data] no patches after patchify")
            self.model.train()
            return

        # -- encode pixel patches -> latent patches
        def _encode_pixel_patches(patches_P2thw: torch.Tensor) -> torch.Tensor:
            outs = []
            for i in range(0, patches_P2thw.shape[0], bs):
                chunk = patches_P2thw[i:i+bs]
                x_list = [x for x in chunk]                      # list of [2,pt,ph,pw]
                pairs = self.vae(x_list, op="encode")            # list of (mu, logv) or [mu]
                for pr in pairs:
                    mu = pr[0] if isinstance(pr, (list, tuple)) else pr
                    if mu.dim() == 5 and mu.shape[0] == 1:
                        mu = mu.squeeze(0)                       # [Cz,nt,H',W']
                    outs.append(mu)
            return torch.stack(outs, dim=0)                       # [B,Cz,nt,H',W']

        def _decode_latent_patches(z_Pcnhw: torch.Tensor) -> torch.Tensor:
            outs = []
            for i in range(0, z_Pcnhw.shape[0], bs):
                chunk = z_Pcnhw[i:i+bs]
                z_list = [z.unsqueeze(0) for z in chunk]         # list [1,Cz,nt,H',W']
                dec = self.vae(z_list, op="decode")              # list of [1,2,pt,ph,pw]
                for o in dec:
                    if isinstance(o, (list, tuple)):
                        o = o[0]
                    outs.append(o.squeeze(0))                    # [2,pt,ph,pw]
            return torch.stack(outs, dim=0)                      # [B,2,pt,ph,pw]

        z_patches = _encode_pixel_patches(patches_px).to(torch.float32)        # [B,Cz,nt,H',W']
        _, Cz, nt, Hlat, Wlat = z_patches.shape

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        raw_model = self.accelerator.unwrap_model(self.model)

        # -- loop over fixed t's
        for t_scalar in t_values:
            t_vec = torch.full((B,), float(t_scalar), device=device)            # [B]
            noise = torch.randn_like(z_patches)
            x_t   = (1.0 - _append_dims(t_vec, z_patches.ndim)) * z_patches + \
                    _append_dims(t_vec, z_patches.ndim) * noise

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                u = raw_model(x_t.to(torch.bfloat16),
                            (t_vec * float(self.t_scale)).to(torch.bfloat16)).float()
            x0_lat = x_t - _append_dims(t_vec, z_patches.ndim) * u              # [B,Cz,nt,H',W']

            # decode predicted x0 to pixels for visualization
            x0_px = _decode_latent_patches(x0_lat)                              # [B,2,pt,ph,pw]

            # -- build grid video of magnitudes
            rows = int(vcfg.get("grid_rows", 2))
            cols = int(vcfg.get("grid_cols", max(1, (B + rows - 1) // rows)))
            Tvid = int(x0_px.shape[2])
            frames = []
            for tt in range(Tvid):
                row_tiles = []
                for r in range(rows):
                    col_tiles = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < B:
                            patch = x0_px[idx]                                   # [2,pt,ph,pw]
                            mag = torch.sqrt(torch.clamp(patch[0, tt]**2 + patch[1, tt]**2, min=0.0))
                        else:
                            mag = torch.zeros_like(x0_px[0, 0, tt])
                        col_tiles.append(mag)
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                grid_img = torch.cat(row_tiles, dim=-2)                           # [rows*ph, cols*pw]
                frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

            arr = np.stack(frames, axis=0)                                        # [T,Hgrid,Wgrid]
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")

            if self.accelerator.is_main_process:
                tag = f"val_latent/x0_pred_from_data_t_{t_scalar:.2f}".replace(".", "p")
                wandb.log({tag: vid}, step=self.global_step)

        if use_ema:
            self.ema.restore(self.model)

        self.model.train()

    @torch.no_grad()
    def validation_simple(self, use_ema: bool = False):
        """
        Simple validation using a plain Euler (x0) reverse sampler:
        • linear sigmas from 1 → 0 (inclusive)
        • optional EMA weights (use_ema=True), otherwise raw model
        • generate N unconditional latent patch videos
        • decode w/ VAE and log a 2×4 grid of magnitudes to W&B
        """
        self.model.eval()
        device = self.accelerator.device

        # --- derive latent geometry via dummy encode ---
        vcfg = self.cfg.get("validation", {})
        N  = int(vcfg.get("num_uncond_videos", 8))
        bs = int(vcfg.get("patch_batch", 64))
        pt = int(vcfg.get("patch_t", 7))
        ph = int(vcfg.get("patch_h", 80))
        pw = int(vcfg.get("patch_w", 80))

        dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
        vae_out = self.vae([dummy], op="encode")
        z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
        Cz, nt, Hlat, Wlat = int(z_mu.shape[1]), int(z_mu.shape[2]), int(z_mu.shape[-2]), int(z_mu.shape[-1])

        # --- linear schedule, fp32 ---
        steps = int(self.cfg.get("sampler", {}).get("num_steps", 25))
        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

        # --- optional EMA application (temporary) ---
        unwrapped = self.accelerator.unwrap_model(self.model)
        ema_applied = False
        if use_ema:
            self.ema.apply_to(unwrapped)
            ema_applied = True

        # --- Euler(x0) sampler (state in fp32; net in bf16) ---
        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        def _sample_euler_x0_velocity(net, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
            import time as _time
            x = noise.to(torch.float32)
            B = x.shape[0]
            total = sigmas.numel() - 1
            pbar = tqdm(total=total, desc="Euler(x0)", dynamic_ncols=True, leave=False,
                        disable=not self.accelerator.is_main_process)
            last = _time.perf_counter()
            for i in range(total):
                t = sigmas[i].expand(B)      # [B], fp32
                s = sigmas[i + 1].expand(B)  # [B], fp32

                x_bf16 = x.to(torch.bfloat16)
                t_bf16 = (t * float(self.t_scale)).to(torch.bfloat16)
                u = net(x_bf16, t_bf16).to(torch.float32)  # velocity

                x0_pred = x - _append_dims(t, x.ndim) * u
                ratio   = _append_dims((s / t.clamp_min(1e-8)), x.ndim)
                x = ratio * x + (1.0 - ratio) * x0_pred

                if self.accelerator.is_main_process:
                    now = _time.perf_counter()
                    pbar.set_postfix({'sec/it': f'{(now - last):.3f}', 't': f'{float(t[0]):.6f}'})
                    last = now
                    pbar.update(1)
            if self.accelerator.is_main_process:
                pbar.close()
            return x

        noise = torch.randn(N, Cz, nt, Hlat, Wlat, device=device, dtype=torch.float32)
        raw_model = unwrapped  # call the underlying module directly
        z_samples = _sample_euler_x0_velocity(raw_model, noise, sigmas)  # fp32 x0

        # --- decode latents in chunks ---
        def _decode_latents(z_bcthw: torch.Tensor):
            outs = []
            for i in range(0, z_bcthw.shape[0], bs):
                chunk = z_bcthw[i:i+bs].float()
                z_list = [z.unsqueeze(0) for z in chunk]
                dec = self.vae(z_list, op="decode")
                for o in dec:
                    if isinstance(o, (list, tuple)):
                        o = o[0]
                    outs.append(o.squeeze(0))  # [2,pt,ph,pw]
            return outs

        xhat_list = _decode_latents(z_samples)

        # --- make a 2×4 grid video of magnitudes ---
        rows = int(vcfg.get("grid_rows", 2))
        cols = int(vcfg.get("grid_cols", max(1, (N + rows - 1) // rows)))
        T = int(xhat_list[0].shape[1])
        frames = []
        for t in range(T):
            row_tiles = []
            for r in range(rows):
                col_tiles = []
                for c in range(cols):
                    idx = r * cols + c
                    if idx < len(xhat_list):
                        patch = xhat_list[idx]
                        mag = torch.sqrt(torch.clamp(patch[0, t]**2 + patch[1, t]**2, min=0.0))
                    else:
                        mag = torch.zeros_like(xhat_list[0][0, t])
                    col_tiles.append(mag)
                row_tiles.append(torch.cat(col_tiles, dim=-1))
            grid_img = torch.cat(row_tiles, dim=-2)
            frames.append(_frame_to_uint8(grid_img.unsqueeze(0)))

        import numpy as _np
        arr = _np.stack(frames, axis=0)
        arr = arr[:, None, :, :]
        arr = _np.repeat(arr, 3, axis=1)
        vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")

        if self.accelerator.is_main_process:
            wandb.log({f"val/uncond_patch_grid_{'ema' if use_ema else 'raw'}": vid}, step=self.global_step)

        # if we temporarily applied EMA, restore original weights
        if ema_applied:
            self.ema.restore(unwrapped)

        self.model.train()

    def validation_simple_inv(self, use_ema: bool = False):
        """
        Euler(x0) + pixel-space DC (against *downsampled* recon) with grads only inside DC.

        DC loss is computed **patch-wise** after applying A (FFT->mask->IFFT) to the
        full-frame reconstruction, so each loss term compares:
            patchify(  A( full_decode(z) )  )  vs  patchify( x_zf )

        Sanity checks:
        • Constant-tiles placement image (patch borders visible, correct placement)
        • Global ramp (patchify→blend) + error map (should be seam-free)

        Logs per-chunk triptychs: GT | ZFR | DC.
        """
        import time, math
        import numpy as _np
        import torch
        import torch.nn.functional as F
        from tqdm.auto import tqdm
        from data.deg import MRIDeg

        # ----------------- small helpers -----------------
        def _to_mag(fr_2hw: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(torch.clamp(fr_2hw[0]**2 + fr_2hw[1]**2, min=0.0))

        def _to_mag_from_complex(x_cmplx_thw: torch.Tensor) -> torch.Tensor:
            return x_cmplx_thw.abs()

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        def _pct_to_stride_len(P, pct):
            ov = max(0.0, min(99.0, float(pct))) / 100.0
            return max(1, int(round(P * (1.0 - ov))))

        def _temporal_chunk_starts(T: int, win: int):
            if win >= T: return [0]
            step = max(1, win - 1)  # 1-frame overlap
            arr = list(range(0, T - win + 1, step))
            if arr[-1] != T - win:
                arr.append(T - win)
            return arr

        def _fft2c(x_2thw: torch.Tensor) -> torch.Tensor:
            xr, xi = x_2thw[0], x_2thw[1]
            xc = torch.complex(xr, xi)
            k  = torch.fft.fft2(xc, norm="ortho")
            kc = torch.fft.fftshift(k, dim=(-2, -1))
            return kc  # complex [T,H,W]

        def _ifft2c(kc_thw: torch.Tensor) -> torch.Tensor:
            k  = torch.fft.ifftshift(kc_thw, dim=(-2, -1))
            x  = torch.fft.ifft2(k, norm="ortho")
            return x  # complex [T,H,W]

        # ----- unified, seam-free spatial tiling (row-major, edges anchored) -----
        def _grid_coords_anchored(L: int, p: int, s: int):
            if L <= p:
                return [0]
            n = 1 + math.ceil((L - p) / s)
            starts = []
            for j in range(n):
                y = j * s
                if y + p > L:
                    y = L - p
                if not starts or y != starts[-1]:
                    starts.append(y)
            return starts

        def _make_coords(H: int, W: int, ph: int, pw: int, sh: int, sw: int):
            ys = _grid_coords_anchored(H, ph, sh)
            xs = _grid_coords_anchored(W, pw, sw)
            coords = []
            for y0 in ys:
                y1 = min(y0 + ph, H); s1 = y1 - y0
                for x0 in xs:
                    x1 = min(x0 + pw, W); s2 = x1 - x0
                    coords.append((y0, y1, s1, x0, x1, s2))
            return coords, len(ys), len(xs)

        @torch.no_grad()
        def _spatial_patchify_video(x_2Thw: torch.Tensor, ph: int, pw: int, coords):
            _, T, H, W = x_2Thw.shape
            P = len(coords)
            out = torch.zeros((P, 2, T, ph, pw), dtype=x_2Thw.dtype, device=x_2Thw.device)
            for idx, (y0,y1,s1,x0,x1,s2) in enumerate(coords):
                out[idx, :, :, :s1, :s2] = x_2Thw[:, :, y0:y1, x0:x1]
            return out

        def _patchify_image_differentiable(x_2Thw: torch.Tensor, coords, ph: int, pw: int) -> torch.Tensor:
            """
            Differentiable extractor: [2,T,H,W] -> [P,2,T,ph,pw]
            """
            patches = []
            for (y0,y1,s1,x0,x1,s2) in coords:
                sl = x_2Thw[:, :, y0:y1, x0:x1]          # [2,T,s1,s2] (view)
                pad = (0, pw - s2, 0, ph - s1)           # (W_left=0, W_right, H_top=0, H_bottom)
                p = F.pad(sl, pad)                        # [2,T,ph,pw], differentiable
                patches.append(p)
            return torch.stack(patches, dim=0)            # [P,2,T,ph,pw]

        def _build_wmaps(coords, device, dtype, mode: str = "box"):
            """
            Per-patch weights [s1,s2]; 'box' (ones) => partition-of-unity after normalization.
            """
            wmaps = []
            for (y0,y1,s1,x0,x1,s2) in coords:
                if mode == "hann" and s1 > 1 and s2 > 1:
                    wy = torch.hann_window(s1, periodic=False, dtype=torch.float32, device=device)
                    wx = torch.hann_window(s2, periodic=False, dtype=torch.float32, device=device)
                    w  = (wy[:, None] * wx[None, :]).to(dtype)
                else:
                    w  = torch.ones((s1, s2), dtype=dtype, device=device)
                wmaps.append(w)
            return wmaps

        def _blend_full(patches_P2Tphw: torch.Tensor, H: int, W: int, coords, wmaps):
            """
            Seamless overlap-add: numerator = Σ w·p, denom = Σ w, then divide.
            Differentiable w.r.t. patches.
            """
            device = patches_P2Tphw.device
            P, C, T, ph, pw = patches_P2Tphw.shape
            assert C == 2
            num = torch.zeros((2, T, H, W), dtype=torch.float32, device=device)
            den = torch.zeros((1, T, H, W), dtype=torch.float32, device=device)
            for idx, (y0,y1,s1,x0,x1,s2) in enumerate(coords):
                w = wmaps[idx][None, None, :, :]                   # [1,1,s1,s2]
                p = patches_P2Tphw[idx, :, :, :s1, :s2].to(torch.float32)
                num[:, :, y0:y1, x0:x1] += (p * w)
                den[:, :, y0:y1, x0:x1] += w
            den = torch.where(den == 0, torch.ones_like(den), den)
            return (num / den).to(patches_P2Tphw.dtype)            # [2,T,H,W]

        def _apply_A_twochan(x_2Thw: torch.Tensor, m_THW: torch.Tensor) -> torch.Tensor:
            """
            Apply forward operator A: FFT -> mask -> IFFT, on full frame.
            x_2Thw: [2,T,H,W] (real/imag channels), m_THW: [T,H,W] in {0,1}
            Returns two-channel image-domain result [2,T,H,W].
            """
            x_c = torch.complex(x_2Thw[0], x_2Thw[1])                         # [T,H,W]
            k   = torch.fft.fftshift(torch.fft.fft2(x_c, norm="ortho"), dim=(-2, -1))
            k_m = k * m_THW                                                   # [T,H,W] complex
            x_d = torch.fft.ifft2(torch.fft.ifftshift(k_m, dim=(-2, -1)), norm="ortho")
            return torch.stack((x_d.real.float(), x_d.imag.float()), dim=0)   # [2,T,H,W]

        def _decode_all_patches_with_grad(z_Pcnhw: torch.Tensor, vae, max_bs: int) -> torch.Tensor:
            """
            Decode all patches with grad, micro-batching to control memory.
            Returns [P,2,pt,ph,pw]; grads flow back to z.
            """
            outs = []
            P = z_Pcnhw.shape[0]
            for i in range(0, P, max_bs):
                chunk = z_Pcnhw[i:i+max_bs]
                z_list = [zz.unsqueeze(0) for zz in chunk]     # list of [1,Cz,nt,H',W']
                with torch.autocast(device_type=chunk.device.type, dtype=torch.bfloat16):
                    dec = vae(z_list, op="decode")             # list of [1,2,pt,ph,pw]
                block = []
                for o in dec:
                    if isinstance(o, (list, tuple)): o = o[0]
                    block.append(o.squeeze(0))                 # [2,pt,ph,pw]
                outs.append(torch.stack(block, dim=0).float())
            return torch.cat(outs, dim=0)                      # [P,2,pt,ph,pw]

        # ----------------- config + setup -----------------
        device = self.accelerator.device
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if use_ema:
            self.ema.apply_to(unwrapped)

        vcfg   = self.cfg.get("validation", {})
        ph     = int(vcfg.get("patch_h", 80))
        pw     = int(vcfg.get("patch_w", 80))
        pt     = int(vcfg.get("patch_t", 7))                      # pixel-time chunk
        val_decode_bs = int(vcfg.get("patch_batch", 64))          # old decode batch (no grads)
        dc_decode_bs  = int(vcfg.get("dc_decode_bs", val_decode_bs))  # **with-grad** decode batch (tunable 1..P)
        dc_inner_iters = int(vcfg.get("dc_inner_iters", 3))       # number of inner DC steps (default 3)
        ov_pct = float(vcfg.get("overlap_spatial_pct", 1.0))      # e.g., 5.0 = 5%
        dc_lmb = float(vcfg.get("dc_lambda", 0.3))                # guidance strength
        dc_step_scale = float(vcfg.get("dc_step_scale", 1.0))     # extra multiplier

        # cap any large with-grad decode batch conservatively
        dc_decode_bs = max(1, dc_decode_bs)

        R_default = int(self.cfg.get("deg", {}).get("R", 4))      # keep user's change (R=4)

        # -------- grab one validation video --------
        try:
            it = iter(self.val_dl)
            batch_list = next(it)
            x_true = None
            for item in batch_list:
                x_true = item
                break
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[validation_simple_inv] empty val loader")
            if use_ema:
                self.ema.restore(unwrapped)
            self.model.train()
            return

        x_true = x_true.to(device=device, dtype=torch.float32)
        if x_true.dim() == 3:
            x_true = x_true.unsqueeze(1)
        _, T, H, W = x_true.shape

        sh = _pct_to_stride_len(ph, ov_pct)
        sw = _pct_to_stride_len(pw, ov_pct)
        coords, n1, n2 = _make_coords(H, W, ph, pw, sh, sw)
        P = len(coords)
        wmaps = _build_wmaps(coords, device=device, dtype=torch.float32, mode="box")  # "box" => partition-of-unity

        # ----------------- SANITY A: constant tiles placement -----------------
        if self.accelerator.is_main_process and bool(vcfg.get("debug_patch_placement", True)):
            pt_dbg = 1
            const_tiles = torch.zeros((P, 2, pt_dbg, ph, pw), device=device, dtype=torch.float32)
            for idx, (y0,y1,s1,x0,x1,s2) in enumerate(coords):
                val = (idx + 1) / (P + 1)
                tile = torch.full((s1, s2), val, device=device)
                tile[0:1, :] = 1.0; tile[-1:, :] = 1.0; tile[:, 0:1] = 1.0; tile[:, -1:] = 1.0  # thin border
                const_tiles[idx, 0, 0, :s1, :s2] = tile
            const_full = _blend_full(const_tiles, H, W, coords, wmaps)  # [2,1,H,W]
            wandb.log({"sanity/constant_tiles": wandb.Image(_frame_to_uint8(const_full[0,0].unsqueeze(0)))},
                    step=self.global_step)

            # ----------------- SANITY B: global ramp -----------------
            yy = torch.linspace(0, 1, steps=H, device=device).view(H, 1).expand(H, W)
            xx = torch.linspace(0, 1, steps=W, device=device).view(1, W).expand(H, W)
            ramp = ((yy + xx) * 0.5).to(torch.float32)  # [H,W]
            full = torch.zeros((2, 1, H, W), device=device, dtype=torch.float32)
            full[0, 0] = ramp
            patches_r = _spatial_patchify_video(full, ph, pw, coords)      # [P,2,1,ph,pw]
            recon_r   = _blend_full(patches_r, H, W, coords, wmaps)        # [2,1,H,W]
            err = (recon_r[0,0] - ramp).abs()
            wandb.log({
                "sanity/global_ramp": wandb.Image(_frame_to_uint8(recon_r[0,0].unsqueeze(0))),
                "sanity/global_ramp_error": wandb.Image(_frame_to_uint8((err*10.0).unsqueeze(0))),  # mag ×10
                "sanity/global_ramp_mae": float(err.mean().detach().cpu()),
                "sanity/global_ramp_max": float(err.max().detach().cpu()),
            }, step=self.global_step)

        # -------- latent geometry via dummy encode (pt,ph,pw) --------
        dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
        vae_out = self.vae([dummy], op="encode")
        mu0 = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
        if mu0.dim() == 5 and mu0.shape[0] == 1:
            mu0 = mu0.squeeze(0)
        Cz, nt, Hlat, Wlat = int(mu0.shape[0]), int(mu0.shape[1]), int(mu0.shape[2]), int(mu0.shape[3])

        # -------- linear sigmas 1→0, fp32 --------
        steps = int(self.cfg.get("sampler", {}).get("num_steps", 25))
        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)

        # ----------------- Patch-wise DC helper (3 inner steps by default) -----------------
        def _dc_latent_from_x0_patchwise(
            z_lat_Pcnhw: torch.Tensor,
            x_zf_TcHW: torch.Tensor,        # complex [Tc<=pt, H, W]
            m_THW_pad: torch.Tensor,        # [pt, H, W] in {0,1}
            coords,
            H: int, W: int, ph: int, pw: int, pt: int,
            step_vec_P: torch.Tensor,       # [P]
        ) -> torch.Tensor:

            # Prepare two-channel ZF target (padded to pt) then patchify (no grad needed)
            xzf_2Thw = torch.stack((x_zf_TcHW.real.float(), x_zf_TcHW.imag.float()), dim=0)  # [2,pt,H,W]
            xzf_patches = _patchify_image_differentiable(xzf_2Thw.detach(), coords, ph, pw)  # [P,2,pt,ph,pw] (target)

            z = z_lat_Pcnhw
            for _ in range(dc_inner_iters):
                z = z.detach().requires_grad_(True)

                # (1) decode all patches WITH grad (micro-batched)
                patches_dec = _decode_all_patches_with_grad(z, self.vae, dc_decode_bs)   # [P,2,pt,ph,pw]

                # (2) blend to full complex frame (differentiable)
                C_full = _blend_full(patches_dec, H, W, coords, wmaps)                    # [2,pt,H,W]

                # (3) apply A globally: FFT -> mask -> IFFT (differentiable)
                C_down = _apply_A_twochan(C_full, m_THW_pad)                              # [2,pt,H,W]

                # (4) patchify both sides => 1-to-1 patch loss
                C_down_patches = _patchify_image_differentiable(C_down, coords, ph, pw)   # [P,2,pt,ph,pw]

                # (5) MSE in complex two-channel space
                L = (C_down_patches - xzf_patches).pow(2).mean()

                # (6) ∇z and latent update
                (g,) = torch.autograd.grad(L, z, retain_graph=False, create_graph=False)
                step = (step_vec_P * dc_lmb * dc_step_scale).view(-1, *([1] * (z.ndim - 1)))  # [P,1,1,1,1]
                step = 15.0
                z = (z - step * g)

            return z.detach()

        # ----------------- Euler(x0) with DC per chunk -----------------
        def _sample_euler_x0_with_dc(raw_model, noise: torch.Tensor,
                                    sigmas: torch.Tensor,
                                    x_zf_chunk_cmplx: torch.Tensor,
                                    m_THW_pad: torch.Tensor) -> torch.Tensor:
            """
            noise: [P,Cz,nt,H',W']; x_zf_chunk_cmplx: complex [pt, H, W]
            returns: [P,Cz,nt,H',W'] (x0 after DC-regularized reverse integration)
            """
            x = noise.to(torch.float32)
            B = x.shape[0]
            total = sigmas.numel() - 1

            pbar = tqdm(total=total, desc="Euler(x0)+DC", dynamic_ncols=True, leave=False,
                        disable=not self.accelerator.is_main_process)
            last = time.perf_counter()

            for i in range(total):
                t = sigmas[i].expand(B)
                s = sigmas[i + 1].expand(B)

                # net velocity (no grad; bf16 autocast)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        u = raw_model(x.to(torch.bfloat16),
                                    (t * float(self.t_scale)).to(torch.bfloat16)).to(torch.float32)

                # predictor in latent space
                x0_pred = x - _append_dims(t, x.ndim) * u
                x1_pred = x + (1 - _append_dims(t, x.ndim)) * u

                # DC inner steps: patch-wise loss after A(full)
                step_vec = (t - s).abs()                               # [P]
                with torch.enable_grad():
                    x0_dc = _dc_latent_from_x0_patchwise(
                        x0_pred, x_zf_chunk_cmplx, m_THW_pad, coords, H, W, ph, pw, pt, step_vec
                    )

                z0_t = _append_dims(t, x.ndim) * x0_dc + (1 - _append_dims(t, x.ndim)) * x0_pred
                z1_t = _append_dims(s, x.ndim).sqrt() * x1_pred + (1 - _append_dims(s, x.ndim)).sqrt() * torch.randn_like(x1_pred)

                # Euler(x0) update with DC-corrected x0
                x = (1 - _append_dims(s, x.ndim)) * z0_t + _append_dims(s, x.ndim) * z1_t
                x = x.detach()

                if self.accelerator.is_main_process:
                    now = time.perf_counter()
                    pbar.set_postfix({'sec/it': f'{(now - last):.3f}', 't': f'{float(t[0]):.6f}'})
                    last = now
                    pbar.update(1)

            if self.accelerator.is_main_process:
                pbar.close()
            return x

        # ----------------- temporal chunks -----------------
        starts = _temporal_chunk_starts(T, pt)
        recon_frames = []   # list of [2,H,W] tensors in order
        prev_start, prev_len = None, None

        for t0 in starts:
            t1 = min(t0 + pt, T)
            Tc_valid = int(t1 - t0)
            x_true_chunk = x_true[:, t0:t1]  # [2,Tc_valid,H,W]

            # --- build ZF target (complex) and pad to pt ---
            deg = MRIDeg(H, Tc_valid, R_default)
            mk = getattr(deg, "mask_ky_t", None)
            if callable(mk): mk = mk()
            m_TH = torch.as_tensor(mk, device=device, dtype=torch.float32)
            if m_TH.shape == (H, Tc_valid):
                m_TH = m_TH.t()  # [Tc_valid, H]
            m_THW = m_TH[:, :, None].expand(Tc_valid, H, W)  # [Tc_valid,H,W]

            kc_true = _fft2c(x_true_chunk)                      # complex [Tc_valid,H,W]
            kc_meas = kc_true * m_THW
            x_zf = _ifft2c(kc_meas)                             # complex [Tc_valid,H,W]

            # pad ZF and mask in time to pt if needed
            if Tc_valid < pt:
                pad = pt - Tc_valid
                zpad_c = torch.zeros((pad, H, W), device=device, dtype=x_zf.dtype)
                x_zf_chunk = torch.cat([x_zf, zpad_c], dim=0)            # [pt,H,W] complex
                m_pad = torch.cat([m_THW, torch.zeros((pad, H, W), device=device)], dim=0)  # [pt,H,W]
            else:
                x_zf_chunk = x_zf
                m_pad = m_THW

            # --- reverse sampling in latent space with DC ---
            noise = torch.randn((P, Cz, nt, Hlat, Wlat), device=device, dtype=torch.float32)
            z_x0 = _sample_euler_x0_with_dc(unwrapped, noise, sigmas, x_zf_chunk, m_pad)  # [P,Cz,nt,H',W']

            # --- decode patches and seamless assemble (no grad) ---
            with torch.no_grad():
                # micro-batch (no-grad) decode for logging
                outs = []
                for i in range(0, z_x0.shape[0], val_decode_bs):
                    chunk = z_x0[i:i+val_decode_bs]
                    z_list = [z.unsqueeze(0) for z in chunk]
                    dec = self.vae(z_list, op="decode")      # list [1,2,pt,ph,pw]
                    for o in dec:
                        if isinstance(o, (list, tuple)): o = o[0]
                        outs.append(o.squeeze(0))            # [2,pt,ph,pw]
                patches_dec = torch.stack(outs, dim=0)       # [P,2,pt,ph,pw]
                xhat_full   = _blend_full(patches_dec, H, W, coords, wmaps)  # [2,pt,H,W]

            # --- log triptych GT|ZFR|DC for the valid frames in this chunk ---
            if self.accelerator.is_main_process:
                frames = []
                fps = int(self.cfg.get("logging", {}).get("latent_grid_fps", 7))
                for tt in range(Tc_valid):
                    gt_mag = _to_mag(x_true_chunk[:, tt]).unsqueeze(0)     # [1,H,W]
                    zf_mag = _to_mag_from_complex(x_zf[tt]).unsqueeze(0)   # [1,H,W]
                    dc_mag = _to_mag(xhat_full[:, tt]).unsqueeze(0)        # [1,H,W]

                    gt_u8 = _frame_to_uint8(gt_mag)
                    zf_u8 = _frame_to_uint8(zf_mag)
                    dc_u8 = _frame_to_uint8(dc_mag)

                    trip = _np.concatenate([gt_u8, zf_u8, dc_u8], axis=-1) # [H, 3W]
                    frames.append(trip)

                arr = _np.stack(frames, axis=0)  # [T, H, 3W]
                arr = arr[:, None, :, :]
                arr = _np.repeat(arr, 3, axis=1)
                vid = wandb.Video(arr, fps=fps, format="mp4")
                wandb.log({f"val/dc_chunk_triptych_{t0:04d}": vid}, step=self.global_step)

                try:
                    # patchify GT (2,Tc,H,W) -> [P,2,Tc,ph,pw]
                    gt_patches = _spatial_patchify_video(x_true_chunk, ph, pw, coords)[:, :, :Tc_valid]
                    # patchify DC recon for valid frames
                    dc_patches = _spatial_patchify_video(xhat_full[:, :Tc_valid], ph, pw, coords)
                    # patchify ZF (build 2-ch first)
                    zf_2Thw = torch.stack((x_zf.real.float(), x_zf.imag.float()), dim=0)  # [2,Tc,H,W]
                    zf_patches = _spatial_patchify_video(zf_2Thw, ph, pw, coords)

                    # magnitude tensors
                    gt_mag = torch.sqrt(gt_patches[:, 0]**2 + gt_patches[:, 1]**2)  # [P,Tc,ph,pw]
                    zf_mag = torch.sqrt(zf_patches[:, 0]**2 + zf_patches[:, 1]**2)  # [P,Tc,ph,pw]
                    dc_mag = torch.sqrt(dc_patches[:, 0]**2 + dc_patches[:, 1]**2)  # [P,Tc,ph,pw]

                    # score patches by coverage above a small floor (use GT)
                    px_thr_plot = float(vcfg.get("plot_px_thr", 0.05))
                    plot_min_frac = float(vcfg.get("plot_min_frac", 0.1))  # ignore near-empty patches
                    frac = (gt_mag > px_thr_plot).float().mean(dim=(1, 2, 3))  # [P]

                    # pick top-K non-empty
                    K = int(vcfg.get("num_patch_compare", 4))
                    K = max(1, min(K, P))
                    nonzero = torch.nonzero(frac > plot_min_frac, as_tuple=False).flatten()
                    if nonzero.numel() > 0:
                        f_sel = frac[nonzero]
                        topk = torch.topk(f_sel, k=min(K, f_sel.numel()), largest=True).indices
                        idxs = nonzero[topk]
                    else:
                        idxs = torch.arange(K, device=device)

                    # build a mosaic using the center frame
                    t_center = min(Tc_valid // 2, Tc_valid - 1)
                    rows = []
                    for gidx in idxs.tolist():
                        gt_u8 = _frame_to_uint8(gt_mag[gidx, t_center].unsqueeze(0))
                        zf_u8 = _frame_to_uint8(zf_mag[gidx, t_center].unsqueeze(0))
                        dc_u8 = _frame_to_uint8(dc_mag[gidx, t_center].unsqueeze(0))
                        row = np.concatenate([gt_u8, zf_u8, dc_u8], axis=1)  # [ph, 3*pw]
                        rows.append(row)
                    mosaic = np.concatenate(rows, axis=0)  # [K*ph, 3*pw]

                    if self.accelerator.is_main_process:
                        wandb.log({"val/patch_triptych4_center": wandb.Image(mosaic)}, step=self.global_step)

                except Exception as e:
                    if self.accelerator.is_main_process:
                        print("[warn] patch-level comparison failed:", e)

            # --- stitch chunk back into sequence (temporal overlap = 1 frame) ---
            if prev_start is None:
                for f in range(Tc_valid):
                    recon_frames.append(xhat_full[:, f])
                prev_start, prev_len = t0, Tc_valid
            else:
                overlap = max(0, (prev_start + prev_len) - t0)
                start_f = min(overlap, Tc_valid)
                for f in range(start_f, Tc_valid):
                    recon_frames.append(xhat_full[:, f])
                prev_start, prev_len = t0, Tc_valid

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # process only the first chunk by default (keep runtime reasonable)
            break

        # ----------------- final full-video logging -----------------
        with torch.no_grad():
            recon = torch.stack(recon_frames, dim=1)  # [2,T',H,W] (T' ≤ T)
            if self.accelerator.is_main_process:
                frames = []
                Tprime = recon.shape[1]
                for t in range(Tprime):
                    mag = _to_mag(recon[:, t]).unsqueeze(0)
                    frames.append(_frame_to_uint8(mag))
                arr = _np.stack(frames, axis=0)
                arr = arr[:, None, :, :]
                arr = _np.repeat(arr, 3, axis=1)
                vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
                wandb.log({"val/recon_dc": vid}, step=self.global_step)

        if use_ema:
            self.ema.restore(unwrapped)

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
    ds_name = ds_cfg["name"]
    DS = try_import_dataset(ds_name)
    dataset = DS(**ds_cfg.get("args", {}))
    bsz = dl_cfg["train_batch_size"] if is_train else dl_cfg["val_batch_size"]
    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=dl_cfg.get("shuffle", True) if is_train else False,
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        drop_last=is_train,
        # IMPORTANT: default collate for train now (no ragged lists)
        collate_fn=None if is_train else ragged_collate,
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

    # Optional pretrained / resume-aware load (mirrors VAE semantics)
    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))
    resume_flag = bool(cfg["model"].get("resume", False))

    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")

        # If *not* resuming and EMA exists, load EMA weights; otherwise load raw model
        print(f"[FM] loaded non-EMA weights from {pretrained_path}")
        state = ckpt.get("model", ckpt)

        # strip possible wrappers
        new_sd = OrderedDict((k[10:] if k.startswith("_orig_mod.") else k, v) for k, v in state.items())
        missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
        if not strict_load:
            print(f"[FM] missing={len(missing)} unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FM] total number of model parameters: {n_params/1e9}B")

    trainer = LatentFMTrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/flow_matching.yaml")
    args = ap.parse_args()
    main(args.config)
