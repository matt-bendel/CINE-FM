#!/usr/bin/env python3
import os, sys, math, time, argparse, importlib, yaml
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

# ---------- import path ----------
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# Data
from data.cine_fm_dataset import CINEFlowMatchDataset
# Sampler (your UniPC flow-matching)
from sampler.flowmatch_unipc import sample_unipc

# ---------- Accelerate ----------
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# -------------------- patch helpers (verbatim) --------------------
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
        S = ceil((D - P) / (n_patches - 1)); S = max(S, 1)
    return S, n_patches

def compute_strides_and_N(data_shape, patch_size=(80,80,11), extra_patch_num=(0,0,0)):
    strides, n_patches_list = [], []
    for D, P, extra in zip(data_shape, patch_size, extra_patch_num):
        S, n_patches = compute_stride_and_n_patches(D, P, extra)
        strides.append(S); n_patches_list.append(n_patches)
    N = torch.prod(torch.tensor(n_patches_list))
    return tuple(strides), N.item(), tuple(n_patches_list)

def patchify(data: torch.Tensor, patch_size):
    data_shape = data.shape
    strides, N, n_list = compute_strides_and_N(data_shape, patch_size)
    S0, S1, S2 = strides; P0, P1, P2 = patch_size
    n0, n1, n2 = n_list
    patches = torch.zeros((N, P0, P1, P2), dtype=data.dtype, device=data.device)
    idx = 0
    for i in range(n0):
        s0 = i * S0; e0 = min(s0 + P0, data_shape[0]); ps0 = slice(0, e0 - s0)
        for j in range(n1):
            s1 = j * S1; e1 = min(s1 + P1, data_shape[1]); ps1 = slice(0, e1 - s1)
            for k in range(n2):
                s2 = k * S2; e2 = min(s2 + P2, data_shape[2]); ps2 = slice(0, e2 - s2)
                patch = data[s0:e0, s1:e1, s2:e2]
                padded = torch.zeros((P0, P1, P2), dtype=data.dtype, device=data.device)
                padded[ps0, ps1, ps2] = patch
                patches[idx] = padded; idx += 1
    return patches, strides

def depatchify(patches: torch.Tensor, data_shape: tuple, patch_size: tuple, strides: tuple):
    D0, D1, D2 = data_shape; P0, P1, P2 = patch_size; S0, S1, S2 = strides
    device = patches.device; dtype = patches.dtype
    out_num = torch.zeros(data_shape, dtype=dtype, device=device)
    out_den = torch.zeros(data_shape, dtype=torch.float32, device=device)
    n0 = max(1, ceil((D0 - P0) / S0) + 1)
    n1 = max(1, ceil((D1 - P1) / S1) + 1)
    n2 = max(1, ceil((D2 - P2) / S2) + 1)
    expected_N = n0 * n1 * n2
    if patches.shape[0] != expected_N:
        raise ValueError(f"N={patches.shape[0]} but expected {expected_N} (= {n0}*{n1}*{n2})")
    O0, O1, O2 = max(0, P0-S0), max(0, P1-S1), max(0, P2-S2)
    def axis_w(L_eff, idx, n, O):
        has_prev = idx > 0; has_next = idx < n-1
        L_left  = min(O if has_prev else 0, L_eff)
        L_right = min(O if has_next else 0, L_eff)
        if L_left + L_right > L_eff:
            if L_left > 0 and L_right > 0:
                tot = L_left + L_right
                L_left_new = max(1, int(round(L_eff * (L_left / tot))))
                L_right_new = L_eff - L_left_new
                L_left, L_right = L_left_new, L_right_new
            else:
                L_left = min(L_left, L_eff); L_right = L_eff - L_left
        w = torch.ones(L_eff, dtype=torch.float32, device=device)
        if L_left > 0:
            w[:L_left] = torch.linspace(0.0, 1.0, steps=L_left, device=device) if L_left > 1 else 0.5
        if L_right > 0:
            w[-L_right:] = torch.linspace(1.0, 0.0, steps=L_right, device=device) if L_right > 1 else 0.5
        return w
    idx = 0
    for i in range(n0):
        s0 = i * S0; e0 = min(s0 + P0, D0); ps0 = slice(0, e0 - s0); w0 = axis_w(e0 - s0, i, n0, O0)
        for j in range(n1):
            s1 = j * S1; e1 = min(s1 + P1, D1); ps1 = slice(0, e1 - s1); w1 = axis_w(e1 - s1, j, n1, O1)
            for k in range(n2):
                s2 = k * S2; e2 = min(s2 + P2, D2); ps2 = slice(0, e2 - s2); w2 = axis_w(e2 - s2, k, n2, O2)
                w = (w0[:, None, None] * w1[None, :, None] * w2[None, None, :])
                patch = patches[idx][ps0, ps1, ps2]
                out_num[s0:e0, s1:e1, s2:e2] += patch * w.to(dtype)
                out_den[s0:e0, s1:e1, s2:e2] += w
                idx += 1
    out_den[out_den == 0] = 1.0
    return out_num / out_den.to(dtype)

# -------------------- utils --------------------
def ragged_collate(batch):  # keep lists (varied T)
    return batch

def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters(): p.requires_grad_(flag)

class EMA:
    def __init__(self, model, decay=0.999):
        self.m = model
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def step(self):
        i = 0
        for p in self.m.parameters():
            if not p.requires_grad: continue
            self.shadow[i].mul_(self.decay).add_(p.data, alpha=1-self.decay); i += 1
    @torch.no_grad()
    def copy_to(self, target):
        i = 0
        for p in target.parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.shadow[i]); i += 1

# -------------------- VAE encode/decode on patches --------------------
@torch.no_grad()
def vae_encode_patches(vae, patches_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """patches_list: list of [2,t,h,w] -> returns list of mu [Cz,n,H',W']"""
    out = []
    BS = 32
    for i in range(0, len(patches_list), BS):
        chunk = patches_list[i:i+BS]
        xhats, mus, logvs, zs = vae(chunk)  # your VAE returns lists aligned to input
        for mu in mus:
            out.append(mu.squeeze(0))
    return out

@torch.no_grad()
def vae_decode_patches(vae, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
    xs = []
    BS = 32
    for i in range(0, len(z_list), BS):
        chunk = [z.unsqueeze(0) for z in z_list[i:i+BS]]
        xs_batch = vae.decode(chunk)
        xs.extend([xb.squeeze(0) for xb in xs_batch])
    return xs

# -------------------- small viz helpers --------------------
@torch.no_grad()
def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0: return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

@torch.no_grad()
def _to_wandb_video_one(x_mag_1t: torch.Tensor, fps: int = 7):
    T = int(x_mag_1t.shape[1])
    frames = [ _frame_to_uint8(x_mag_1t[:, t]) for t in range(T) ]
    arr = np.stack(frames, axis=0)[:, None, :, :]
    arr = np.repeat(arr, 3, axis=1)
    return wandb.Video(arr, fps=fps, format="mp4")

# -------------------- dataloaders --------------------
def build_dataloader(ds_cfg: Dict[str, Any], dl_cfg: Dict[str, Any], split: str) -> DataLoader:
    dataset = CINEFlowMatchDataset(**ds_cfg["args"], split=split)
    bsz = dl_cfg["train_batch_size"] if split == "train" else dl_cfg["val_batch_size"]
    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=(split == "train") and dl_cfg.get("shuffle", True),
        num_workers=dl_cfg.get("num_workers", 8),
        pin_memory=dl_cfg.get("pin_memory", False),
        drop_last=(split == "train"),
        collate_fn=ragged_collate,
    )

# -------------------- validation helpers --------------------
@torch.no_grad()
def _rand_odd_T(t_list: List[int]) -> int:
    # t_list should already be odd numbers between 1..11
    i = torch.randint(low=0, high=len(t_list), size=(1,)).item()
    return int(t_list[i])

@torch.no_grad()
def _sample_random_clip(x: torch.Tensor, L: int) -> torch.Tensor:
    # x: [2,T,H,W], returns [2,L,H,W]
    _, T, _, _ = x.shape
    if L == 1:
        t0 = T // 2
        return x[:, t0:t0+1]
    if T >= L:
        start = torch.randint(low=0, high=T-L+1, size=(1,)).item()
        return x[:, start:start+L]
    # circular if too short
    start = torch.randint(low=0, high=T, size=(1,)).item()
    idxs = [(start + i) % T for i in range(L)]
    return x[:, idxs]

# -------------------- main train+validate --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, required=True, help="CardiacVAE checkpoint path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------- Accelerate setup ----------
    proj_cfg = ProjectConfiguration(project_dir=cfg["logging"]["out_dir"])
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True)
    accelerator = Accelerator(project_config=proj_cfg, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    torch.backends.cudnn.benchmark = True

    # ---------- WandB (main only) ----------
    wdir  = cfg["logging"].get("wandb_dir", None)
    wcache= cfg["logging"].get("wandb_cache_dir", None)
    if accelerator.is_main_process:
        if wdir:   os.environ["WANDB_DIR"] = str(wdir)
        if wcache: os.environ["WANDB_CACHE_DIR"] = str(wcache)
        wandb.init(
            project=cfg["logging"]["project"],
            name=cfg["logging"].get("run_name", "latent-fm"),
            config=cfg,
            dir=wdir,
        )
    accelerator.wait_for_everyone()

    # ---------- Data ----------
    train_dl = build_dataloader(cfg["train_dataset"], cfg["dataloader"], "train")
    val_dl   = build_dataloader(cfg["val_dataset"],   cfg["dataloader"], "val")

    # ---------- VAE (frozen)
    VAECls = dynamic_import(cfg["vae"]["import_path"], cfg["vae"]["class_name"])
    vae = VAECls(**cfg["vae"]["args"]).to(device).eval()
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if accelerator.is_main_process:
        print(f"[VAE] loaded {args.vae_ckpt} (missing={len(missing)} unexpected={len(unexpected)})")
    set_requires_grad(vae, False)

    # ---------- FM model
    FMCls = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    fm = FMCls(**cfg["model"]["args"])
    fm = fm.to(device)

    opt = torch.optim.AdamW(
        fm.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"]
    )
    ema = EMA(fm, decay=cfg["optim"].get("ema_decay", 0.999))

    # prepare w/ Accelerate
    fm, opt, train_dl, val_dl = accelerator.prepare(fm, opt, train_dl, val_dl)

    total_steps = int(cfg["optim"]["total_steps"])
    accum_steps = int(cfg["optim"].get("accum_steps", 1))
    log_every   = int(cfg["logging"]["log_every_steps"])
    val_every   = int(cfg["logging"]["val_every_steps"])
    save_every  = int(cfg["logging"]["save_every_steps"])
    out_dir     = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    global_step = 0
    pbar = tqdm(total=total_steps, disable=not accelerator.is_main_process, dynamic_ncols=True, desc="train_fm")

    # -------------- training loop (step-based) --------------
    while global_step < total_steps:
        for batch in train_dl:
            with accelerator.accumulate(fm):
                # batch is a ragged list; for train split, each item may be a LIST of clips -> flatten
                clips: List[torch.Tensor] = []
                for item in batch:
                    if isinstance(item, list): clips.extend(item)
                    else: clips.append(item)

                # ---- patchify & VAE-encode to latents; put P in batch dim ----
                patch_h = int(cfg["validation"]["patch_h"])
                patch_w = int(cfg["validation"]["patch_w"])
                z_data_list: List[torch.Tensor] = []

                for x in clips:
                    x = x.to(device=device, dtype=torch.float32)  # [2,T,H,W]
                    _, T, H, W = x.shape
                    tsize = 11 if T > 1 else 1
                    patch_size = (tsize, patch_h, patch_w)

                    r_patches, _ = patchify(x[0], patch_size)  # [P,t,h,w]
                    i_patches, _ = patchify(x[1], patch_size)
                    Pn = int(r_patches.shape[0])
                    patches_list = [ torch.stack((r_patches[n], i_patches[n]), dim=0) for n in range(Pn) ]
                    mus = vae_encode_patches(vae, patches_list)  # list of [Cz,n,H',W']
                    z_data_list.extend([m.to(device) for m in mus])

                if len(z_data_list) == 0:
                    continue

                z_data = torch.stack(z_data_list, dim=0)  # [B=P_total, Cz, n, H', W']
                B = z_data.shape[0]
                z0 = torch.randn_like(z_data)
                t  = torch.rand(B, device=device) * (1.0 - 1e-3) + 1e-3  # (0,1]
                xt = (1.0 - t)[:, None, None, None, None] * z0 + t[:, None, None, None, None] * z_data
                v_target = z_data - z0

                fm.train()
                v_pred = fm(xt, t)
                loss = F.mse_loss(v_pred, v_target)

                # backward + step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg["optim"].get("grad_clip", 0.0) > 0:
                        accelerator.clip_grad_norm_(fm.parameters(), cfg["optim"]["grad_clip"])
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.step()

                # step accounting
                if accelerator.sync_gradients:
                    global_step += 1
                    if accelerator.is_main_process:
                        pbar.update(1)
                        pbar.set_postfix(loss=float(loss.detach().cpu()))

                    if global_step % log_every == 0 and accelerator.is_main_process:
                        wandb.log({"train/loss": float(loss.detach().cpu()),
                                   "train/batch_latent_patches": B}, step=global_step)

                    if global_step % val_every == 0:
                        metrics = validate(accelerator, cfg, vae, fm, global_step, val_dl)
                        if accelerator.is_main_process:
                            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

                    if global_step % save_every == 0 and accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(fm)
                        path = os.path.join(out_dir, f"fm_step_{global_step:07d}.pt")
                        torch.save({"model": unwrapped.state_dict(), "cfg": cfg, "step": global_step}, path)
                        wandb.save(path)
                        print(f"[ckpt] saved {path}")

                    if global_step >= total_steps:
                        break

        if global_step >= total_steps:
            break

    if accelerator.is_main_process:
        pbar.close()
        print("Training complete.")

# -------------------- validation --------------------
@torch.no_grad()
def validate(accelerator: Accelerator, cfg, vae, fm, step: int, val_dl: DataLoader) -> Dict[str, float]:
    device = accelerator.device
    fm.eval()

    # ---- velocity MSE over some val batches ----
    val_num_batches = int(cfg["logging"].get("val_num_batches", 8))
    t_choices = cfg.get("train_dataset", {}).get("args", {}).get("t_choices", [1,3,5,7,9,11])
    patch_h = int(cfg["validation"]["patch_h"]); patch_w = int(cfg["validation"]["patch_w"])

    mse_vals = []
    it = iter(val_dl)
    for _ in range(val_num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        clips: List[torch.Tensor] = []
        for item in batch:
            if isinstance(item, list):
                clips.extend(item)
            else:
                L = _rand_odd_T(t_choices)
                clips.append(_sample_random_clip(item.to(device), L))

        z_data_list: List[torch.Tensor] = []
        for x in clips:
            x = x.to(device=device, dtype=torch.float32)
            _, T, H, W = x.shape
            tsize = 11 if T > 1 else 1
            patch_size = (tsize, patch_h, patch_w)
            r_patches, _ = patchify(x[0], patch_size)
            i_patches, _ = patchify(x[1], patch_size)
            Pn = int(r_patches.shape[0])
            patches_list = [ torch.stack((r_patches[n], i_patches[n]), dim=0) for n in range(Pn) ]
            mus = vae_encode_patches(vae, patches_list)
            z_data_list.extend([m.to(device) for m in mus])

        if len(z_data_list) == 0:
            continue

        z_data = torch.stack(z_data_list, dim=0)
        B = z_data.shape[0]
        z0 = torch.randn_like(z_data)
        t  = torch.rand(B, device=device) * (1.0 - 1e-3) + 1e-3
        xt = (1.0 - t)[:, None, None, None, None] * z0 + t[:, None, None, None, None] * z_data
        v_target = z_data - z0
        v_pred = fm(xt, t)
        mse = F.mse_loss(v_pred, v_target)
        mse_vals.append(mse.detach())

    if len(mse_vals) == 0:
        mse_mean = torch.tensor(0.0, device=device)
    else:
        mse_stack = torch.stack(mse_vals)
        mse_mean = mse_stack.mean()

    # gather across GPUs
    g = accelerator.gather_for_metrics(mse_mean)
    fm_mse = float(g.mean().detach().cpu())

    # ---- Full diffusion → 11 pixel frames (main process only) ----
    out = {"fm_mse": fm_mse}

    if accelerator.is_main_process:
        try:
            val_ds = CINEFlowMatchDataset(**cfg["val_dataset"]["args"], split="val")
            if len(val_ds) > 0:
                x = val_ds[0].to(device=device, dtype=torch.float32)  # [2,T,H,W]
                _, T, H, W = x.shape
                if T >= 11: x = x[:, :11]
                else:
                    idxs = list(range(T)) + [i % T for i in range(11-T)]
                    x = x[:, idxs]
                patch_size = (11, patch_h, patch_w)

                r_patches, strides = patchify(x[0], patch_size)
                i_patches, _ = patchify(x[1], patch_size)
                Np = int(r_patches.shape[0])

                # latent shape probe
                probe_mu = vae_encode_patches(vae, [torch.stack((r_patches[0], i_patches[0]), dim=0)])[0]
                Cz, nL, Hh, Ww = probe_mu.shape

                # sample in latent space with UniPC
                noise = torch.randn(Np, Cz, nL, Hh, Ww, device=device)
                K = int(cfg["sampler"].get("num_steps", 18))
                sigmas = torch.logspace(0.0, math.log10(1e-3), steps=K, device=device)

                # Wrap fm to (x, t) -> v; UniPC passes vector t
                def fm_wrap(z, t):  # z: [B,...], t: [B]
                    return fm(z, t)

                z_samples = sample_unipc(fm_wrap, noise, sigmas, extra_args={}, disable=True,
                                         variant=cfg["sampler"].get("variant", "bh1"))
                # z_samples is (last denoised); shape [Np, Cz, nL, Hh, Ww]
                z_list = [ z_samples[i] for i in range(Np) ]
                xhat_patches = vae_decode_patches(vae, z_list)  # list of [2,11,h,w]

                r_rec = torch.stack([p[0] for p in xhat_patches], dim=0)
                i_rec = torch.stack([p[1] for p in xhat_patches], dim=0)
                xhat_r = depatchify(r_rec, (11, H, W), patch_size, strides)
                xhat_i = depatchify(i_rec, (11, H, W), patch_size, strides)
                xhat   = torch.stack((xhat_r, xhat_i), dim=0)  # [2,11,H,W]

                xm  = torch.sqrt(x[0]**2 + x[1]**2).unsqueeze(0)   # [1,11,H,W]
                xhm = torch.sqrt(xhat[0]**2 + xhat[1]**2).unsqueeze(0)

                # log videos
                wandb.log({
                    "vis/val_video_gt":    _to_wandb_video_one(xm),
                    "vis/val_video_recon": _to_wandb_video_one(xhm),
                }, step=step)

                # simple mag L1
                out["mag_l1"] = float((xm - xhm).abs().mean().item())
        except Exception as e:
            print(f"[val] sampling/logging skipped: {e}")

    fm.train()
    return out

# -------------------- entrypoint --------------------
if __name__ == "__main__":
    main()
