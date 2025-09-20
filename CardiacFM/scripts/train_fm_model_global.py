#!/usr/bin/env python3
import os, sys, time, math, importlib, yaml
from typing import Dict, Any, List, Tuple, Optional

# ------------------------------------------------------------------------------
# Path + imports
# ------------------------------------------------------------------------------
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# EMA
from utils.ema import Ema

# ------------------------------------------------------------------------------
# Utilities: dynamic import, ragged collate
# ------------------------------------------------------------------------------
def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def try_import_dataset(name: str):
    # Expect raw pixel datasets at train/val time (we build latents on-the-fly).
    if name in ("CINEDataset", "CINEFlowMatchDataset"):
        mod = importlib.import_module("data.cine_dataset")
        return getattr(mod, "CINEDataset")
    elif name == "CINEFlowMatchLatentDataset":
        # Supported, but we still assemble global latents here; items must carry (H,W) meta or be uniform.
        mod = importlib.import_module("data.cine_flow_dataset")
        return getattr(mod, "CINEFlowMatchLatentDataset")
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

def ragged_collate(batch):
    # Return list-of-items; we handle variable sizes in compute_loss().
    return batch

# ------------------------------------------------------------------------------
# Small viz helpers (for val videos)
# ------------------------------------------------------------------------------
def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

# ------------------------------------------------------------------------------
# Overlap math (5% overlap) and spatial tiling
# ------------------------------------------------------------------------------
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
    return coords, n1, n2

def _axis_weights(L_eff: int, idx: int, n: int, O: int, device) -> torch.Tensor:
    """
    Smooth overlap-add weights (linear ramps in the overlap).
    """
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

# ------------------------------------------------------------------------------
# Pixel patchify (spatial only, full time clip of length pt)
# ------------------------------------------------------------------------------
@torch.no_grad()
def spatial_patchify_video(x_2thw: torch.Tensor, ph: int, pw: int, sh: int, sw: int, coords) -> torch.Tensor:
    """
    x: [2, pt, H, W] -> [P, 2, pt, ph, pw] (zero-padded at borders)
    """
    device = x_2thw.device; dtype = x_2thw.dtype
    _, T, H, W = x_2thw.shape
    P = len(coords)
    out = torch.zeros((P, 2, T, ph, pw), dtype=dtype, device=device)
    for idx, (y0,y1,s1,x0,x1,s2, *_rest) in enumerate(coords):
        patch = torch.zeros((2, T, ph, pw), dtype=dtype, device=device)
        patch[:, :, :s1, :s2] = x_2thw[:, :, y0:y1, x0:x1]
        out[idx] = patch
    return out

# ------------------------------------------------------------------------------
# VAE encode (pixel patches -> latent patches)
# ------------------------------------------------------------------------------
@torch.no_grad()
def encode_pixel_patches_to_latents(vae, patches_P2tHW: torch.Tensor, bs_vae: int) -> torch.Tensor:
    """
    Input:  [P, 2, pt, ph, pw]
    Output: [P, Cz, nt, H', W']   (nt is ~4 when pt=7)
    """
    outs = []
    for i in range(0, patches_P2tHW.shape[0], bs_vae):
        chunk = patches_P2tHW[i:i+bs_vae]                # [B,2,pt,ph,pw]
        x_list = [x for x in chunk]                      # list of [2,pt,ph,pw]
        pairs = vae(x_list, op="encode")                 # list of (mu, logv) or mu
        for pr in pairs:
            mu = pr[0] if isinstance(pr, (list, tuple)) else pr
            if mu.dim() == 5 and mu.shape[0] == 1:       # squeeze potential batch dim
                mu = mu.squeeze(0)
            outs.append(mu)                              # [Cz, nt, H', W']
    return torch.stack(outs, dim=0)

# ------------------------------------------------------------------------------
# Latent overlap-add → global latent
# ------------------------------------------------------------------------------
@torch.no_grad()
def depatchify_latents_over_space(
    z_patches: torch.Tensor,              # [P, Cz, nt, H', W']
    n1: int, n2: int,
    stride_h_lat: int, stride_w_lat: int,
) -> torch.Tensor:
    """
    Assemble a **global latent** from latent patches using the same 5% overlap logic
    (performed in latent coordinates).
    """
    device = z_patches.device; dtype = z_patches.dtype
    P, Cz, nt, hpatch, wpatch = z_patches.shape
    assert P == n1 * n2, f"P={P} but n1*n2={n1*n2}"

    H_lat = (n1 - 1) * stride_h_lat + hpatch
    W_lat = (n2 - 1) * stride_w_lat + wpatch

    out_num = torch.zeros((Cz, nt, H_lat, W_lat), dtype=dtype, device=device)
    out_den = torch.zeros((1, 1, H_lat, W_lat), dtype=torch.float32, device=device)

    O1 = max(0, hpatch - stride_h_lat)
    O2 = max(0, wpatch - stride_w_lat)

    idx = 0
    for j in range(n1):
        y0 = j * stride_h_lat; y1 = y0 + hpatch
        w1 = _axis_weights(hpatch, j, n1, O1, device)
        for k in range(n2):
            x0 = k * stride_w_lat; x1 = x0 + wpatch
            w2 = _axis_weights(wpatch, k, n2, O2, device)
            w = (w1[None, None, :, None] * w2[None, None, None, :])  # [1,1,hpatch,wpatch]
            p = z_patches[idx]                                       # [Cz,nt,hpatch,wpatch]
            out_num[:, :, y0:y1, x0:x1] += (p * w).to(out_num.dtype)
            out_den[:, :, y0:y1, x0:x1] += w
            idx += 1

    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)                        # [Cz, nt, H_lat, W_lat]

# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------
class LatentFMTrainer:
    """
    Train on **global latents** constructed on-the-fly:
      • spatial 5% overlap in pixel → encode each patch via VAE → overlap-add in **latent**
      • pt (pixel frames) → nt latent frames (e.g., 7→4) inferred from a dummy encode
      • model is the updated global-latent transformer: input [B,Cz,nt,H',W'] → velocity
    """
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model = model
        # Optional compile
        if bool(cfg.get("compile", False)):
            self.model = torch.compile(self.model, fullgraph=False)

        # schedule scaling used consistently everywhere
        self.t_scale = float(cfg.get("sampler", {}).get("t_scale", 1000.0))

        proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)
        self.accelerator = Accelerator(project_config=proj_cfg, kwargs_handlers=[ddp_kwargs], mixed_precision="bf16")

        opt_cfg = cfg["optim"]
        self.total_steps = int(opt_cfg["total_steps"])
        self.accum_steps = int(opt_cfg.get("accum_steps", 1))
        self.grad_clip   = float(opt_cfg.get("grad_clip", 0.0))

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
        items = [self.model, self.optimizer, self.train_dl, self.val_dl]
        if self.scheduler is not None:
            items.append(self.scheduler)
            (self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler) = self.accelerator.prepare(*items)
        else:
            (self.model, self.optimizer, self.train_dl, self.val_dl) = self.accelerator.prepare(*items)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema = Ema(unwrapped, decay=float(self.cfg["optim"].get("ema_decay", 0.999)))

        # W&B init
        wdir  = self.cfg["logging"].get("wandb_dir", None)
        wcache = self.cfg["logging"].get("wandb_cache_dir", None)
        if wdir:   os.environ["WANDB_DIR"] = str(wdir)
        if wcache: os.environ["WANDB_CACHE_DIR"] = str(wcache)
        if self.accelerator.is_main_process:
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"].get("run_name", "latent_fm_global"),
                config=cfg,
                dir=wdir,
            )

        # -------- Validation/patch config (shared with training) --------
        vcfg = self.cfg.get("validation", {})
        self.patch_h  = int(vcfg.get("patch_h", 80))
        self.patch_w  = int(vcfg.get("patch_w", 80))
        self.patch_t  = int(vcfg.get("patch_t", 7))        # pixel frames consumed by encoder
        self.patch_bs = int(vcfg.get("patch_batch", 64))   # VAE microbatch
        self.overlap_pct = float(vcfg.get("overlap_pct", 5.0))
        self.val_num_samples = int(vcfg.get("num_uncond_videos", 1))

        # ---------- VAE (encode/decode) ----------
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
                print(f"[VAE] loaded weights from {vae_ckpt}")
        self.vae = self.vae.to(self.accelerator.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # ---------- Infer latent channel + nt via dummy encode ----------
        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.patch_t, self.patch_h, self.patch_w,
                                device=self.accelerator.device, dtype=torch.float32)
            vae_out = self.vae([dummy], op="encode")
            z_mu = vae_out[0][0] if isinstance(vae_out[0], (list, tuple)) else vae_out[0]
            if z_mu.dim() == 5 and z_mu.shape[0] == 1:
                z_mu = z_mu.squeeze(0)
            self.Cz  = int(z_mu.shape[0])
            self.nt  = int(z_mu.shape[1])       # should be 4 when patch_t=7
            self.hlp = int(z_mu.shape[2])       # latent patch height
            self.wlp = int(z_mu.shape[3])       # latent patch width

        if self.accelerator.is_main_process:
            print(f"[latent] Cz={self.Cz}  nt={self.nt}  latent_patch=({self.hlp},{self.wlp}) from pt={self.patch_t}")

    # ----------------- Global latent builder (one clip) -----------------
    @torch.no_grad()
    def build_global_latent_from_clip(self, x_2thw: torch.Tensor) -> torch.Tensor:
        """
        x_2thw: [2, T, H, W]; we use exactly patch_t frames (center-crop if T>patch_t, pad if T<patch_t).
        Returns Z: [Cz, nt, H_lat, W_lat] (float32)
        """
        device = x_2thw.device
        C, T, H, W = x_2thw.shape
        assert C == 2, f"expected complex [2, T, H, W], got {tuple(x_2thw.shape)}"

        # choose a deterministic temporal window of length patch_t
        pt = self.patch_t
        if T == pt:
            x_win = x_2thw
        elif T > pt:
            s = max(0, (T - pt) // 2)
            x_win = x_2thw[:, s:s+pt]
        else:  # pad at tail with zeros
            pad = pt - T
            pad_tail = torch.zeros((2, pad, H, W), device=device, dtype=x_2thw.dtype)
            x_win = torch.cat([x_2thw, pad_tail], dim=1)

        # spatial tiling on pixels
        ph, pw = self.patch_h, self.patch_w
        sh = pct_to_stride_len(ph, self.overlap_pct)
        sw = pct_to_stride_len(pw, self.overlap_pct)
        coords, n1, n2 = spatial_coords(H, W, ph, pw, sh, sw)

        px_patches = spatial_patchify_video(x_win, ph, pw, sh, sw, coords)      # [P,2,pt,ph,pw]
        z_patches  = encode_pixel_patches_to_latents(self.vae, px_patches, self.patch_bs)  # [P,Cz,nt,H',W']

        # latent overlap-add (5%) in latent coordinates
        sh_lat = pct_to_stride_len(self.hlp, self.overlap_pct)
        sw_lat = pct_to_stride_len(self.wlp, self.overlap_pct)
        Z = depatchify_latents_over_space(z_patches, n1, n2, sh_lat, sw_lat)    # [Cz,nt,H_lat,W_lat]
        return Z.float()

    # ----------------- Rectified-Flow loss (global latent) -----------------
    def _rf_loss_on_global_latent(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [Cz, nt, H', W'] (float32)
        """
        device = self.accelerator.device
        Z = Z.to(device=device, dtype=torch.float32)
        # noise + t in bf16 for model path; loss in fp32
        noise = torch.randn_like(Z)
        t = torch.rand((), device=device, dtype=torch.float32)  # scalar for this sample
        x_t = (1.0 - t) * Z + t * noise
        target = (noise - Z)

        x_in = x_t.unsqueeze(0).to(torch.bfloat16)               # [1,Cz,nt,H',W']
        t_in = (t * self.t_scale).view(1).to(torch.bfloat16)

        pred = self.model(x_in, t_in).float().squeeze(0)        # [Cz,nt,H',W']
        return torch.nn.functional.mse_loss(pred, target)

    # ----------------- Training loop -----------------
    def train(self):
        self.accelerator.print("Starting Flow Matching training on GLOBAL latents (bf16 model)…")
        log_cfg = self.cfg["logging"]

        pbar = tqdm(
            total=self.total_steps,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            desc="train",
            leave=True,
        )

        train_iter = iter(self.train_dl)
        last = time.perf_counter()

        while self.global_step < self.total_steps:
            try:
                batch_list = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch_list = next(train_iter)

            self.model.train()
            with self.accelerator.accumulate(self.model):
                # accumulate scalar loss over ragged items
                losses = []
                for item in batch_list:
                    # accept dict/list/array; extract [2,T,H,W]
                    if isinstance(item, (list, tuple)):
                        x = item[0]
                    elif isinstance(item, dict):
                        x = item.get("video") or item.get("x") or item.get("data")
                        if x is None:
                            continue
                    else:
                        x = item
                    x = x.to(device=self.accelerator.device, dtype=torch.float32)
                    if x.dim() == 3:
                        x = x.unsqueeze(1)  # [2,1,H,W]
                    # Build global latent for this clip
                    with torch.no_grad():
                        Z = self.build_global_latent_from_clip(x)  # [Cz,nt,H',W']
                    loss_i = self._rf_loss_on_global_latent(Z)
                    losses.append(loss_i)

                if not losses:
                    # empty batch; skip
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss = torch.stack(losses).mean()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if self.grad_clip and self.grad_clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    # EMA on unwrapped
                    self.ema.update(self.accelerator.unwrap_model(self.model))

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
                        wandb.log({"train/loss": float(loss.detach().cpu()),
                                   "lr": self.optimizer.param_groups[0]["lr"]},
                                  step=self.global_step)

                    # unconditional val
                    if (self.global_step % log_cfg["val_every_steps"] == 0) and self.global_step > 0:
                        self.validation_uncond()
                        self.accelerator.wait_for_everyone()

                    # checkpoint
                    if (self.global_step % log_cfg["save_every_steps"] == 0) and self.global_step > 0:
                        self.save_checkpoint()
                        self.accelerator.wait_for_everyone()

                    self.global_step += 1

                # (if not sync_gradients, we are in gradient accumulation; loop continues)

            if self.global_step >= self.total_steps:
                break

        if self.accelerator.is_main_process:
            pbar.close()
            self.accelerator.print("Training complete.")

    # ----------------- Unconditional validation -----------------
    @torch.no_grad()
    def validation_uncond(self):
        """
        Pick one val clip, infer its latent global grid, sample **one** global latent with Euler(x0),
        decode via VAE, and log a simple magnitude video to W&B.
        """
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)

        # get one val item
        try:
            val_it = iter(self.val_dl)
            batch_list = next(val_it)
        except StopIteration:
            if self.accelerator.is_main_process:
                print("[val] empty val loader")
            return

        # pull first item
        item = batch_list[0]
        if isinstance(item, (list, tuple)):
            x = item[0]
        elif isinstance(item, dict):
            x = item.get("video") or item.get("x") or item.get("data")
        else:
            x = item
        x = x.to(device=self.accelerator.device, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [2,1,H,W]

        # Build global latent ONCE to get H'/W'
        Z = self.build_global_latent_from_clip(x)  # [Cz,nt,H',W']
        Cz, nt, Hlat, Wlat = Z.shape

        # Euler(x0) schedule 1→0
        steps = int(self.cfg.get("sampler", {}).get("num_steps", 25))
        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=self.accelerator.device, dtype=torch.float32)

        def _append_dims(v: torch.Tensor, target_ndim: int) -> torch.Tensor:
            return v[(...,) + (None,) * (target_ndim - v.ndim)]

        def _sample_euler_x0_velocity(net, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
            x = noise.to(torch.float32)
            B = x.shape[0]
            total = sigmas.numel() - 1
            pbar = tqdm(total=total, desc="val Euler(x0)", dynamic_ncols=True, leave=False,
                        disable=not self.accelerator.is_main_process)
            for i in range(total):
                t = sigmas[i].expand(B)
                s = sigmas[i + 1].expand(B)
                x_bf16 = x.to(torch.bfloat16)
                t_bf16 = (t * float(self.t_scale)).to(torch.bfloat16)
                u = net(x_bf16, t_bf16).to(torch.float32)
                x0_pred = x - _append_dims(t, x.ndim) * u
                ratio   = _append_dims((s / t.clamp_min(1e-8)), x.ndim)
                x = ratio * x + (1.0 - ratio) * x0_pred
                if self.accelerator.is_main_process:
                    pbar.update(1)
            if self.accelerator.is_main_process:
                pbar.close()
            return x

        noise = torch.randn((self.val_num_samples, Cz, nt, Hlat, Wlat),
                            device=self.accelerator.device, dtype=torch.float32)
        z_x0 = _sample_euler_x0_velocity(unwrapped, noise, sigmas)  # [N,Cz,nt,H',W']

        # decode each sample and log first one
        def _decode_latents(z_bcthw: torch.Tensor):
            outs = []
            bs = int(self.cfg.get("validation", {}).get("patch_batch", 64))
            for i in range(0, z_bcthw.shape[0], bs):
                chunk = z_bcthw[i:i+bs].float()
                z_list = [z.unsqueeze(0) for z in chunk]       # [1,Cz,nt,H',W']
                dec = self.vae(z_list, op="decode")            # list of [1,2,pt,ph,pw]
                for o in dec:
                    if isinstance(o, (list, tuple)):
                        o = o[0]
                    outs.append(o.squeeze(0))                  # [2,pt,ph,pw]
            return outs

        xhat_list = _decode_latents(z_x0)
        # single video magnitude
        vid = xhat_list[0]                                     # [2,pt,ph,pw]
        T = int(vid.shape[1])
        frames = []
        for t in range(T):
            mag = torch.sqrt(torch.clamp(vid[0, t]**2 + vid[1, t]**2, min=0.0)).unsqueeze(0)
            frames.append(_frame_to_uint8(mag))
        arr = np.stack(frames, axis=0)[:, None, :, :]
        arr = np.repeat(arr, 3, axis=1)
        grid_vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
        if self.accelerator.is_main_process:
            wandb.log({"val/uncond_global": grid_vid}, step=self.global_step)

        self.model.train()

    # ----------------- Checkpoint -----------------
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

# ------------------------------------------------------------------------------
# Dataloaders
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg.get("seed", 42)))

    # Datasets
    train_dl = build_dataloader(cfg["train_dataset"], cfg["dataloader"], is_train=True)
    val_dl   = build_dataloader(cfg["val_dataset"],   cfg["dataloader"], is_train=False)

    # Model
    ModelClass = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = ModelClass(**cfg["model"]["args"]).to(torch.bfloat16)

    # Optional pretrained load (no lazy layers; safe across H/W)
    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))
    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        # strip possible wrappers
        new_sd = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items() }
        missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
        if not strict_load:
            print(f"[load] missing={len(missing)}  unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FM] total params: {n_params/1e9:.3f}B")

    # Train
    trainer = LatentFMTrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/flow_matching.yaml")
    args = ap.parse_args()
    main(args.config)
