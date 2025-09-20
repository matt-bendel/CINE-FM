#!/usr/bin/env python3
import os, sys, math, time, importlib, yaml, warnings
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

# ---------- path ----------
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# ---------- core deps ----------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import wandb

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# ---------- ours ----------
from utils.ema import Ema

# ============================================================================
# small utils
# ============================================================================

def dynamic_import(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    return getattr(mod, class_name)

def try_import_dataset(name: str):
    """
    TRAIN: CINEFlowMatchLatentDataset (precomputed latent patches with 5% overlap)
    VAL:   CINEDataset  (raw pixel videos) or dataset returning one val sample
    """
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

# ============================================================================
# tiling / coords (PIXEL space)
# ============================================================================

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

# ============================================================================
# latent geometry helpers: map pixel tiling -> latent tiling (and reverse)
# ============================================================================

@torch.no_grad()
def infer_latent_geometry_from_vae(vae, pt: int, ph: int, pw: int, device) -> Tuple[int,int,int,int,float,float]:
    """
    Returns: (Cz, nt, Hlp, Wlp, sy, sx) for the given pixel patch size (pt,ph,pw).
    nt is the VAE latent time (e.g., 4 when encoding 7 frames).
    sy,sx ~ latent_per_pixel scale factors (Hlp/ph, Wlp/pw).
    """
    dummy = torch.zeros(1, 2, pt, ph, pw, device=device, dtype=torch.float32)
    out = vae([dummy], op="encode")
    mu = out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
    if mu.dim() == 5 and mu.shape[0] == 1:
        mu = mu.squeeze(0)
    Cz, nt, Hlp, Wlp = int(mu.shape[0]), int(mu.shape[1]), int(mu.shape[2]), int(mu.shape[3])
    sy = Hlp / float(ph); sx = Wlp / float(pw)
    return Cz, nt, Hlp, Wlp, sy, sx

def pixel_to_latent_coords(coords_px, sy: float, sx: float, Hlg: int, Wlg: int):
    """
    Map pixel patch coords -> latent patch coords on a global latent grid.
    Each entry becomes (ly0, ly1, ls1, lx0, lx1, ls2, j, k, n1, n2)
    """
    lat_coords = []
    for (y0,y1,s1,x0,x1,s2,j,k,n1,n2) in coords_px:
        ly0 = int(round(y0 * sy)); ly1 = int(round(y1 * sy))
        lx0 = int(round(x0 * sx)); lx1 = int(round(x1 * sx))
        ly0 = max(0, min(ly0, Hlg)); ly1 = max(0, min(ly1, Hlg))
        lx0 = max(0, min(lx0, Wlg)); lx1 = max(0, min(lx1, Wlg))
        ls1 = max(0, ly1 - ly0);     ls2 = max(0, lx1 - lx0)
        lat_coords.append((ly0, ly1, ls1, lx0, lx1, ls2, j, k, n1, n2))
    return lat_coords

def choose_grid_from_P(P: int) -> Tuple[int,int]:
    """
    Fallback when dataset doesn't give H/W: most-square (n1,n2) factorization with n1*n2=P.
    """
    best = (1, P)
    best_diff = P - 1
    r = int(math.sqrt(P))
    for n1 in range(1, r+1):
        if P % n1 == 0:
            n2 = P // n1
            if abs(n2 - n1) < best_diff:
                best = (n1, n2)
                best_diff = abs(n2 - n1)
    return best  # (rows, cols)

@torch.no_grad()
def assemble_global_latent(
    z_patches: torch.Tensor,  # [Cz, nt, P, Hlp, Wlp]
    n1: int, n2: int,
    sh_lat: int, sw_lat: int,
) -> Tuple[torch.Tensor, List[Tuple[int,int,int,int,int,int,int,int,int,int]]]:
    """
    Overlap-add latent patches into a global latent (reversible).
    Returns:
      z_global: [Cz, nt, Hlg, Wlg]
      lat_coords: latent-space coords per patch (for later re-patching)
    """
    device = z_patches.device; dtype = z_patches.dtype
    Cz, nt, P, Hlp, Wlp = z_patches.shape
    assert P == n1 * n2, f"grid {n1}x{n2} ≠ P={P}"

    Hlg = (n1 - 1) * sh_lat + Hlp
    Wlg = (n2 - 1) * sw_lat + Wlp

    out_num = torch.zeros((Cz, nt, Hlg, Wlg), dtype=dtype, device=device)
    out_den = torch.zeros((1,   1,  Hlg, Wlg), dtype=torch.float32, device=device)

    O1 = max(0, Hlp - sh_lat); O2 = max(0, Wlp - sw_lat)

    lat_coords = []
    idx = 0
    for j in range(n1):
        ly0 = j * sh_lat; ly1 = min(ly0 + Hlp, Hlg); ls1 = ly1 - ly0
        w1 = _axis_weights(ls1, j, n1, O1, device)
        for k in range(n2):
            lx0 = k * sw_lat; lx1 = min(lx0 + Wlp, Wlg); ls2 = lx1 - lx0
            w2 = _axis_weights(ls2, k, n2, O2, device)
            w = (w1[None, None, :, None] * w2[None, None, None, :])  # [1,1,ls1,ls2]

            p = z_patches[:, :, idx, :ls1, :ls2]                     # [Cz,nt,ls1,ls2]
            out_num[:, :, ly0:ly1, lx0:lx1] += (p * w).to(out_num.dtype)
            out_den[:, :, ly0:ly1, lx0:lx1] += w
            lat_coords.append((ly0, ly1, ls1, lx0, lx1, ls2, j, k, n1, n2))
            idx += 1

    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    z_global = out_num / out_den.to(out_num.dtype)
    return z_global, lat_coords

@torch.no_grad()
def repatchify_from_global_latent(
    z_global: torch.Tensor,           # [Cz, nt, Hlg, Wlg]
    lat_coords: List[Tuple[int,int,int,int,int,int,int,int,int,int]],
    Hlp: int, Wlp: int
) -> torch.Tensor:
    """
    Extract latent patches back from a global latent (zeros-pad edges).
    Returns: [P, Cz, nt, Hlp, Wlp]
    """
    device = z_global.device; dtype = z_global.dtype
    Cz, nt, Hlg, Wlg = z_global.shape
    P = len(lat_coords)
    out = torch.zeros((P, Cz, nt, Hlp, Wlp), dtype=dtype, device=device)
    for idx, (ly0,ly1,ls1,lx0,lx1,ls2, *_rest) in enumerate(lat_coords):
        if ls1 > 0 and ls2 > 0:
            out[idx, :, :, :ls1, :ls2] = z_global[:, :, ly0:ly1, lx0:lx1]
    return out

# ============================================================================
# pixel-space overlap-add (for validation decode)
# ============================================================================

@torch.no_grad()
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

# ============================================================================
# tiny viz
# ============================================================================

def _frame_to_uint8(img_1hw: torch.Tensor, lo_p=0.01, hi_p=0.99) -> np.ndarray:
    f = torch.nan_to_num(img_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo = torch.quantile(flat, lo_p); hi = torch.quantile(flat, hi_p)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)
    return g.numpy()

# ============================================================================
# dataloader
# ============================================================================

def build_dataloader(ds_cfg: Dict[str, Any], dl_cfg: Dict[str, Any], is_train: bool) -> DataLoader:
    DS = try_import_dataset(ds_cfg["name"])
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

# ============================================================================
# trainer
# ============================================================================

class LatentGlobalTrainer:
    """
    Train the transformer on **global latents** assembled from **overlapping latent patches (5%)**.
    Dataset items can be:
      • torch.Tensor         : [Cz, nt, P, Hlp, Wlp]
      • dict with 'z' tensor : same shape + optional meta:
            - 'hw'          : (H_px, W_px) of the pixel FOV used for patching
            - 'grid_shape'  : (n1_rows, n2_cols) for the patch grid
    """
    def __init__(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict[str, Any]):
        cfg['logging']['out_dir'] = os.path.join(cfg['logging']['out_dir'], "flowmatch_global")
        self.cfg = cfg
        self.model = model

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

        # wandb setup
        wdir  = self.cfg["logging"].get("wandb_dir", None)
        wcache = self.cfg["logging"].get("wandb_cache_dir", None)
        if wdir:   os.environ["WANDB_DIR"] = str(wdir)
        if wcache: os.environ["WANDB_CACHE_DIR"] = str(wcache)

        if self.accelerator.is_main_process:
            wandb.init(
                project=cfg["logging"]["project"],
                name=cfg["logging"].get("run_name", "latent_fm_global_bf16"),
                config=cfg,
                dir=wdir,
            )

        # prepare (Accelerate)
        prep_objs = [self.model, self.optimizer, self.train_dl, self.val_dl]
        if self.scheduler is not None:
            prep_objs.append(self.scheduler)
            (self.model, self.optimizer, self.train_dl, self.val_dl, self.scheduler) = self.accelerator.prepare(*prep_objs)
        else:
            (self.model, self.optimizer, self.train_dl, self.val_dl) = self.accelerator.prepare(*prep_objs)

        # EMA on unwrapped module
        self.ema = Ema(self.accelerator.unwrap_model(self.model), decay=float(opt_cfg.get("ema_decay", 0.999)))

        # --- Validation VAE ---
        vcfg = self.cfg.get("validation", {})
        self.val_patch_h  = int(vcfg.get("patch_h", 80))
        self.val_patch_w  = int(vcfg.get("patch_w", 80))
        self.val_patch_t  = int(vcfg.get("patch_t", 7))
        self.val_patch_bs = int(vcfg.get("patch_batch", 64))

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

        # latent geometry cache for val (pt-> dims)
        Cz, nt, Hlp, Wlp, sy, sx = infer_latent_geometry_from_vae(self.vae, self.val_patch_t, self.val_patch_h, self.val_patch_w, self.accelerator.device)
        self.latent_geom_val = dict(Cz=Cz, nt=nt, Hlp=Hlp, Wlp=Wlp, sy=sy, sx=sx)

    # ----------------------------------------------------------------------
    # loss on GLOBAL latents (Rectified-Flow / velocity regression)
    # ----------------------------------------------------------------------
    def _rf_loss_global(self, Zg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Zg: [N, Cz, nt, Hlg, Wlg] (bf16)
        """
        device = Zg.device
        N = Zg.shape[0]

        noise  = torch.randn_like(Zg)                                # bf16
        t_uni  = torch.rand((N,), device=device, dtype=torch.bfloat16)

        # x_t and target
        t_b = t_uni.view(N, *([1] * (Zg.ndim - 1)))                  # bf16
        x_t = (1.0 - t_b) * Zg + t_b * noise                         # bf16
        target = (noise - Zg)                                        # bf16

        # model forward (bf16, time scaled)
        t_inp = (t_uni.to(torch.float32) * self.t_scale).to(torch.bfloat16)
        pred  = self.model(x_t, t_inp)                               # bf16, shape = Zg

        # MSE in fp32
        mse = torch.nn.functional.mse_loss(pred.float(), target.float())
        return {"total": mse, "mse": mse}

    # ----------------------------------------------------------------------
    # batch -> GLOBAL latents (5% overlap)
    # ----------------------------------------------------------------------
    def _item_to_global_latent(
        self,
        item: Any,
        ph: int, pw: int, overlap_pct: float = 5.0,
        assume_row_major: bool = True
    ) -> torch.Tensor:
        """
        Convert a dataset item (patch latents) to a single GLOBAL latent.
        Accepts:
          • torch.Tensor: [Cz, nt, P, Hlp, Wlp]
          • dict: {'z': tensor, 'hw': (H,W), 'grid_shape': (n1,n2)}
        Returns: z_global [Cz, nt, Hlg, Wlg] (dtype bf16).
        """
        device = self.accelerator.device

        # --- unpack holder ---
        if isinstance(item, dict):
            z = item.get("z", None)
            hw = item.get("hw", None)
            grid_shape = item.get("grid_shape", None)
        else:
            z, hw, grid_shape = item, None, None

        if not torch.is_tensor(z):
            raise RuntimeError("Item must contain latent 'z' tensor.")
        z = z.to(device=device, dtype=torch.bfloat16)  # [Cz, nt, P, Hlp, Wlp]
        Cz, nt, P, Hlp, Wlp = int(z.shape[0]), int(z.shape[1]), int(z.shape[2]), int(z.shape[3]), int(z.shape[4])

        # latent stride derived from **latent patch size** and 5% overlap
        sh_lat = pct_to_stride_len(Hlp, overlap_pct)
        sw_lat = pct_to_stride_len(Wlp, overlap_pct)

        # --- derive (n1,n2) ---
        if grid_shape is not None:
            n1, n2 = int(grid_shape[0]), int(grid_shape[1])
            if n1 * n2 != P:
                warnings.warn(f"[grid_shape] {grid_shape} != P={P}; attempting fallback.")
                n1, n2 = choose_grid_from_P(P)
        elif hw is not None:
            # Use pixel H/W to deduce (n1,n2) deterministically
            H_px, W_px = int(hw[0]), int(hw[1])
            sh_px = pct_to_stride_len(ph, overlap_pct)
            sw_px = pct_to_stride_len(pw, overlap_pct)
            n1 = max(1, math.ceil((H_px - ph) / sh_px) + 1)
            n2 = max(1, math.ceil((W_px - pw) / sw_px) + 1)
            if n1 * n2 != P:
                warnings.warn(f"[hw→grid] computed {n1}x{n2} != P={P}; falling back to most-square factorization.")
                n1, n2 = choose_grid_from_P(P)
        else:
            # last resort (assumes row-major enumeration)
            n1, n2 = choose_grid_from_P(P)

        z_global, _lat_coords = assemble_global_latent(z, n1, n2, sh_lat, sw_lat)  # [Cz, nt, Hlg, Wlg]
        return z_global

    # ----------------------------------------------------------------------
    # train step (list-of-items -> batch of GLOBAL latents)
    # ----------------------------------------------------------------------
    def _compute_loss(self, batch_list: List[Any]) -> Dict[str, torch.Tensor]:
        vcfg = self.cfg.get("validation", {})
        ph = int(vcfg.get("patch_h", 80))
        pw = int(vcfg.get("patch_w", 80))
        overlap_pct = 5.0

        Zg_list = []
        for item in batch_list:
            if isinstance(item, list):
                # dataset may yield a list of latent-clip tensors for a video
                for z in item:
                    Zg_list.append(self._item_to_global_latent(z, ph, pw, overlap_pct))
            else:
                Zg_list.append(self._item_to_global_latent(item, ph, pw, overlap_pct))

        if not Zg_list:
            zero = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
            return {"total": zero, "mse": zero}

        Zg = torch.stack(Zg_list, dim=0)  # [N, Cz, nt, Hlg, Wlg] (bf16)
        return self._rf_loss_global(Zg)

    # ----------------------------------------------------------------------
    def train(self):
        log_cfg = self.cfg["logging"]
        opt_cfg = self.cfg["optim"]

        self.accelerator.print("Starting Flow Matching on GLOBAL latents (bf16)…")

        pbar = tqdm(
            total=self.total_steps,
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            desc="train",
            leave=True,
        )
        train_iter = iter(self.train_dl)

        while self.global_step < self.total_steps:
            try:
                batch_list = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                batch_list = next(train_iter)

            with self.accelerator.accumulate(self.model):
                losses = self._compute_loss(batch_list)
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

                    if self.accelerator.is_main_process and (self.global_step % log_cfg["log_every_steps"] == 0):
                        scalars = {f"train/{k}": float(v.detach().cpu()) for k, v in losses.items() if torch.is_tensor(v)}
                        scalars["lr"] = self.optimizer.param_groups[0]["lr"]
                        wandb.log(scalars, step=self.global_step)

                    if (self.global_step % log_cfg["val_every_steps"] == 0) and (self.global_step > 0):
                        if self.accelerator.is_main_process:
                            pbar.write(f"[val] step {self.global_step}")
                        self.validate_uncond()  # ← unconditional only
                        self.accelerator.wait_for_everyone()

                    if (self.global_step % log_cfg["save_every_steps"] == 0) and (self.global_step > 0):
                        if self.accelerator.is_main_process:
                            pbar.write(f"[ckpt] step {self.global_step}")
                        self.save_checkpoint()
                        self.accelerator.wait_for_everyone()

                    self.global_step += 1
                    pbar.update(1)

        if self.accelerator.is_main_process:
            pbar.close()
            self.accelerator.print("Training complete.")

    # ----------------------------------------------------------------------
    # unconditional validation on ONE val sample:
    #   noise -> global latent -> model Euler(x0) -> re-patchify (latent)
    #   -> VAE decode per patch -> pixel OLA -> W&B video
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def validate_uncond(self):
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema.apply_to(unwrapped)

        device = self.accelerator.device
        vcfg   = self.cfg.get("validation", {})
        pt     = int(vcfg.get("patch_t", 7))
        ph     = int(vcfg.get("patch_h", 80))
        pw     = int(vcfg.get("patch_w", 80))
        bs_vae = int(vcfg.get("patch_batch", 64))
        steps  = int(self.cfg.get("sampler", {}).get("num_steps", 25))

        # ----- latent geometry for this (pt,ph,pw)
        Cz, nt, Hlp, Wlp, sy, sx = (self.latent_geom_val["Cz"], self.latent_geom_val["nt"],
                                    self.latent_geom_val["Hlp"], self.latent_geom_val["Wlp"],
                                    self.latent_geom_val["sy"], self.latent_geom_val["sx"])

        # ----- pull one val item to learn pixel H,W and grid (deterministic)
        try:
            val_it = iter(self.val_dl)
            vitem_batch = next(val_it)
            # choose the first element of the ragged batch
            x_any = vitem_batch[0] if isinstance(vitem_batch, list) else vitem_batch
            # Accept either a raw pixel video [2,T,H,W] or dict with 'video'
            if isinstance(x_any, dict):
                x_true = x_any.get("video") or x_any.get("x") or x_any.get("data")
            else:
                x_true = x_any
            x_true = x_true.to(device=device, dtype=torch.float32)
            if x_true.dim() == 3:
                x_true = x_true.unsqueeze(1)
            _, T, H, W = x_true.shape
        except Exception:
            # fallback: pretend a default FOV to derive a grid
            H, W = 256, 256
            T = pt

        # pixel tiling & coords
        sh_px = pct_to_stride_len(ph, 5.0)
        sw_px = pct_to_stride_len(pw, 5.0)
        coords_px, n1, n2 = spatial_coords(H, W, ph, pw, sh_px, sw_px)
        P = n1 * n2

        # latent tiling stride and global latent size
        sh_lat = pct_to_stride_len(Hlp, 5.0)
        sw_lat = pct_to_stride_len(Wlp, 5.0)
        Hlg = (n1 - 1) * sh_lat + Hlp
        Wlg = (n2 - 1) * sw_lat + Wlp

        # map pixel coords -> latent coords (not strictly needed for Euler; needed for re-patchify)
        coords_lat = pixel_to_latent_coords(coords_px, sy, sx, Hlg, Wlg)

        # ----- Euler(x0) on GLOBAL latent -----
        sigmas = torch.linspace(1.0, 0.0, steps=steps + 1, device=device, dtype=torch.float32)
        B = 1
        x = torch.randn(B, Cz, nt, Hlg, Wlg, device=device, dtype=torch.float32)
        total = sigmas.numel() - 1
        for i in tqdm(range(total), desc="val Euler(x0)", disable=not self.accelerator.is_main_process, leave=False):
            t = sigmas[i].expand(B)
            s = sigmas[i + 1].expand(B)

            xb = x.to(torch.bfloat16)
            tb = (t * float(self.t_scale)).to(torch.bfloat16)
            u  = self.model(xb, tb).to(torch.float32)  # velocity on global latent

            x0 = x - t.view(B, *([1] * (x.ndim - 1))) * u
            ratio = (s / t.clamp_min(1e-8)).view(B, *([1] * (x.ndim - 1)))
            x = ratio * x + (1.0 - ratio) * x0

        z_global = x[0].to(torch.float32)  # [Cz, nt, Hlg, Wlg]

        # ----- RE-PATCHIFY (latent) -> [P,Cz,nt,Hlp,Wlp] -----
        z_patches = repatchify_from_global_latent(z_global, coords_lat, Hlp, Wlp)  # [P,Cz,nt,Hlp,Wlp]

        # ----- VAE decode per patch -> pixel patches -----
        # z_list as [1,Cz,nt,Hlp,Wlp] items for the VAE
        patches_pix = []
        for i in range(0, P, bs_vae):
            chunk = z_patches[i:i+bs_vae].float()  # VAE decode in fp32
            z_list = [z.unsqueeze(0) for z in chunk]
            dec = self.vae(z_list, op="decode")    # list of [1,2,pt,ph,pw]
            for o in dec:
                if isinstance(o, (list, tuple)):
                    o = o[0]
                patches_pix.append(o.squeeze(0))   # [2,pt,ph,pw]
        patches_pix = torch.stack(patches_pix, dim=0)  # [P,2,pt,ph,pw]

        # ----- assemble full video in pixel space (OLA) -----
        x_full = depatchify2d_over_time(patches_pix, H, W, ph, pw, sh_px, sw_px, coords_px)  # [2,pt,H,W]

        # ----- log a quick magnitude video -----
        frames = []
        Tvis = int(x_full.shape[1])
        for t in range(Tvis):
            mag = torch.sqrt(torch.clamp(x_full[0, t]**2 + x_full[1, t]**2, min=0.0)).unsqueeze(0)
            frames.append(_frame_to_uint8(mag))
        arr = np.stack(frames, axis=0)[:, None, :, :]
        arr = np.repeat(arr, 3, axis=1)
        vid = wandb.Video(arr, fps=int(self.cfg.get("logging", {}).get("latent_grid_fps", 7)), format="mp4")
        if self.accelerator.is_main_process:
            wandb.log({"val/uncond_global_sample": vid}, step=self.global_step)

        self.ema.restore(unwrapped)
        self.model.train()

    # ----------------------------------------------------------------------
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

# ============================================================================
# main
# ============================================================================

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg.get("seed", 42)))

    train_dl = build_dataloader(cfg["train_dataset"], cfg["dataloader"], is_train=True)
    val_dl   = build_dataloader(cfg["val_dataset"],   cfg["dataloader"], is_train=False)

    ModelClass = dynamic_import(cfg["model"]["import_path"], cfg["model"]["class_name"])
    model = ModelClass(**cfg["model"]["args"]).to(torch.bfloat16)

    # optional pretrained / resume-aware load (no "lazy" modules touched here)
    pretrained_path = cfg["model"].get("load_state_dict_from", None)
    strict_load = bool(cfg["model"].get("strict_load", False))
    resume_flag = bool(cfg["model"].get("resume", False))

    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if (not resume_flag) and ("ema" in ckpt):
            print(f"[FM] loading EMA weights from {pretrained_path}")
            state = ckpt["ema"]
        else:
            print(f"[FM] loading raw weights from {pretrained_path}")
            state = ckpt.get("model", ckpt)
        new_sd = OrderedDict((k[10:] if k.startswith("_orig_mod.") else k, v) for k, v in state.items())
        missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
        if not strict_load:
            print(f"[FM] missing={len(missing)} unexpected={len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[FM] total parameters: {n_params/1e9:.3f}B")

    trainer = LatentGlobalTrainer(model, train_dl, val_dl, cfg)
    trainer.train()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/fm_h100.yaml")
    args = ap.parse_args()
    main(args.config)
