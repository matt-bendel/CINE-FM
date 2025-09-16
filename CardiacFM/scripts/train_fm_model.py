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
from sampler.flowmatch_unipc import sample_unipc_x


# ==================== FramePack-style time utilities ====================

def flux_time_shift(t: torch.Tensor, mu: float = 1.15, sigma: float = 1.0) -> torch.Tensor:
    """
    FramePack-style time reparam: t∈[0,1], higher mu concentrates steps near t=1.
    """
    t = torch.clamp(t, min=1e-5, max=1.0)
    emu = math.e ** float(mu)
    return emu / (emu + (1.0 / t - 1.0).pow(sigma))

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

def get_flux_sigmas_from_mu(n_steps: int, mu: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Produce (n_steps+1) schedule 1→0 (inclusive), then reparam via flux_time_shift.
    """
    t = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device, dtype=dtype)
    return flux_time_shift(t, mu=mu)


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
        mod = importlib.import_module("data.cine_fm_latent_dataset")
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

        self.t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))  # <<< consistent t scaling

        proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True)
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

        # prepare under accelerate
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
    def compute_loss(self, batch_list: List[Any]) -> Dict[str, torch.Tensor]:
        """
        TRAIN batches: list-of-items; each item is a *list* of z clips [Cz,7,P,H',W'].
        Flatten P across items → [N, Cz, 7, H', W'] (bf16).
        """
        device = self.accelerator.device
        z_batches = []
        for item in batch_list:
            if isinstance(item, list):
                for z in item:
                    z = z.to(device=device, dtype=torch.bfloat16)   # <<< ensure bf16 inputs
                    z = z.permute(2, 0, 1, 3, 4).contiguous()       # [P,Cz,7,H',W']
                    z_batches.append(z)
        if not z_batches:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return {"total": zero, "mse": zero}
        Z = torch.cat(z_batches, dim=0)  # [N,Cz,7,H',W'] bf16
        return self._fm_loss(Z)

    # -------- training loop --------
    def train(self):
        log_cfg = self.cfg["logging"]
        opt_cfg = self.cfg["optim"]

        self.accelerator.print("Starting Flow Matching training (bf16)…")

        pbar = tqdm(
            total=self.total_steps,
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
                        self.validate()
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
        Unconditional validation (no GT):
          • infer (T,H,W) from first val item (shape only)
          • temporal stride = T-1 (1-frame overlap), 50% spatial overlap
          • sample latents (bf16) with UniPC (t scaled by 1000)
          • decode with VAE (fp32) → overlap-add full synthetic video
          • log sample and a patch-grid video
        """
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema.apply_to(unwrapped)

        try:
            val_iter = iter(self.val_dl)
            batch_list = next(val_iter)
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[val] empty val loader")
            self.ema.restore(unwrapped); self.model.train(); return

        ref = None
        for it in batch_list:
            if torch.is_tensor(it):
                ref = it; break
        if ref is None:
            if self.accelerator.is_main_process:
                print("[val] unexpected val batch; skip")
            self.ema.restore(unwrapped); self.model.train(); return

        _, Ttot, H, W = ref.shape

        pt = int(self.val_patch_t)   # default 7
        ph = int(self.val_patch_h)
        pw = int(self.val_patch_w)
        st = max(1, pt - 1)          # 1-frame temporal overlap
        sh = max(1, ph // 2)         # 50% spatial overlap
        sw = max(1, pw // 2)
        t_starts = temporal_starts_full_coverage(Ttot, pt, st)

        # latent spatial size via dummy encode (fp32 VAE)
        dummy = torch.zeros(1, 2, pt, ph, pw, device=self.accelerator.device, dtype=torch.float32)
        mu, _ = self.vae([dummy], op="encode")
        Cz, Hlat, Wlat = int(mu[0].shape[1]), int(mu[0].shape[-2]), int(mu[0].shape[-1])

        # grid dimensions
        def _n12(H, W, ph, pw, sh, sw):
            n1 = max(1, math.ceil((H - ph) / sh) + 1)
            n2 = max(1, math.ceil((W - pw) / sw) + 1)
            return n1, n2, n1 * n2
        n1, n2, P = _n12(H, W, ph, pw, sh, sw)

        # sampler schedule
        scfg   = self.cfg.get("sampler", {})
        steps  = int(scfg.get("num_steps", 18))
        shift  = scfg.get("shift", None)      # if set: mu = ln(shift)
        sigma_k = float(scfg.get("sigma_exponent", 1.0))

        seq_len = int(pt * Hlat * Wlat)
        if shift is None:
            mu = calculate_flux_mu(
                seq_len,
                x1=float(scfg.get("x1", 256)),
                y1=float(scfg.get("y1", 0.5)),
                x2=float(scfg.get("x2", 4096)),
                y2=float(scfg.get("y2", 1.15)),
                exp_max=float(scfg.get("exp_max", 7.0)),
            )
        else:
            mu = math.log(float(shift))

        sigmas = get_flux_sigmas_from_mu(steps, mu, device=self.accelerator.device, dtype=torch.bfloat16)
        if sigma_k != 1.0:
            sigmas = sigmas.pow(sigma_k)

        # wrapper to apply t_scale and bf16 inside sampler
        def model_bf16_tscaled(x, t, **kwargs):
            # x,t from sampler; ensure bf16 and scale time
            x_bf16 = x.to(torch.bfloat16)
            t_bf16 = (t.to(torch.bfloat16) * self.t_scale)
            return self.model(x_bf16, t_bf16, **kwargs)

        # ---- sample & decode patches for each temporal window ----
        dec_patches_r, dec_patches_i = [], []

        def _decode_latents(z_bchw_t):
            # decode with VAE in fp32 for safety
            outs = []
            bs = max(1, self.val_patch_bs)
            for i in range(0, z_bchw_t.shape[0], bs):
                chunk = z_bchw_t[i:i+bs].float()
                z_list = [z.unsqueeze(0) for z in chunk]        # list of [1,Cz,pt,H',W']
                dec = self.vae(z_list, op="decode")             # list of [1,2,pt,ph,pw]
                outs += [o.squeeze(0) for o in dec]
            return outs

        for _t0 in t_starts:
            noise = torch.randn(P, Cz, pt, Hlat, Wlat, device=self.accelerator.device, dtype=torch.bfloat16)
            z_samples = sample_unipc_x(model_bf16_tscaled, noise, sigmas, extra_args=None, disable=True, variant=scfg.get("variant","bh1"))
            xhat_list = _decode_latents(z_samples)                # len=P, each [2,pt,ph,pw]
            r_stack = torch.stack([o[0] for o in xhat_list], dim=0)  # [P,pt,ph,pw]
            i_stack = torch.stack([o[1] for o in xhat_list], dim=0)
            dec_patches_r.append(r_stack)
            dec_patches_i.append(i_stack)

        # ---- temporal overlap-add to full video ----
        xr = torch.zeros((Ttot, H, W), device=self.accelerator.device, dtype=torch.float32)
        xi = torch.zeros_like(xr)
        wr = torch.zeros_like(xr); wi = torch.zeros_like(xr)

        def axis_weights_1d(L_eff, idx, starts, win_len):
            n = len(starts)
            has_prev = idx > 0
            has_next = idx < n - 1
            O = max(0, win_len - (starts[1]-starts[0] if n > 1 else win_len))
            w = torch.ones(L_eff, device=self.accelerator.device, dtype=torch.float32)
            if has_prev and O > 0:
                w[:O] = 0.5 if O == 1 else torch.linspace(0, 1, O, device=self.accelerator.device)
            if has_next and O > 0:
                w[-O:] = 0.5 if O == 1 else torch.linspace(1, 0, O, device=self.accelerator.device)
            return w

        for wi_idx, t0 in enumerate(t_starts):
            w_t = axis_weights_1d(pt, wi_idx, t_starts, pt)
            for dt in range(pt):
                t_abs = t0 + dt
                if t_abs >= Ttot: break
                rP = dec_patches_r[wi_idx][:, dt]   # [P,ph,pw]
                iP = dec_patches_i[wi_idx][:, dt]
                xr[t_abs] += depatchify2d_fixed_stride(rP, (H, W), ph, pw, sh, sw) * w_t[dt]
                xi[t_abs] += depatchify2d_fixed_stride(iP, (H, W), ph, pw, sh, sw) * w_t[dt]
                wr[t_abs] += w_t[dt]; wi[t_abs] += w_t[dt]

        xr = xr / torch.clamp_min(wr, 1e-8)
        xi = xi / torch.clamp_min(wi, 1e-8)
        xhat = torch.stack((xr, xi), dim=0)  # [2,T,H,W]

        # ---- log synthetic sample (magnitude) ----
        xhm = torch.sqrt(torch.clamp(xhat[0]*xhat[0] + xhat[1]*xhat[1], min=0.0)).unsqueeze(0)  # [1,T,H,W]
        media = {"val/uncond_sample_video": _to_wandb_video_one(xhm, fps=self.cfg.get("logging",{}).get("latent_grid_fps", 7))}
        if self.accelerator.is_main_process:
            wandb.log(media, step=self.global_step)

        # ---- also log a grid of patch videos for the first window ----
        def _patch_grid_video(r_stack, i_stack, n1, n2, pt, ph, pw):
            frames = []
            for t in range(pt):
                tiles = []
                for j in range(n1):
                    row = []
                    for k in range(n2):
                        pidx = j * n2 + k
                        mag = torch.sqrt(torch.clamp(r_stack[pidx, t]**2 + i_stack[pidx, t]**2, min=0.0))
                        row.append(mag)
                    tiles.append(torch.cat(row, dim=-1))
                grid = torch.cat(tiles, dim=-2)  # [n1*ph, n2*pw]
                frames.append(_frame_to_uint8(grid.unsqueeze(0)))
            arr = np.stack(frames, axis=0)
            arr = arr[:, None, :, :]
            arr = np.repeat(arr, 3, axis=1)
            return wandb.Video(arr, fps=int(self.cfg.get("logging",{}).get("latent_grid_fps", 7)), format="mp4")

        if len(dec_patches_r) > 0 and self.accelerator.is_main_process:
            wandb.log(
                {"val/patch_grid_win0": _patch_grid_video(dec_patches_r[0], dec_patches_i[0], n1, n2, pt, ph, pw)},
                step=self.global_step
            )

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
    model = ModelClass(**cfg["model"]["args"])

    # Optional resume / init from checkpoint (non-EMA by default)
    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))
    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        new_sd = OrderedDict((k[10:] if k.startswith("_orig_mod.") else k, v) for k, v in state.items())
        missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
        print(f"[FM] loaded pretrained from {pretrained_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FM] total number of model parameters: {n_params}")

    trainer = LatentFMTrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/flowmatch.yaml")
    args = ap.parse_args()
    main(args.config)
