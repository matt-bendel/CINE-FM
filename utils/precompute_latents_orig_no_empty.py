#!/usr/bin/env python3
import os, sys, json, argparse, importlib, random
from typing import Dict, Any, List, Tuple
from collections import OrderedDict
import numpy as np
import h5py
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import yaml

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# ---------------- utils ----------------

def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

@torch.no_grad()
def robust_norm_qmag_global(vol_2thw: torch.Tensor, q=0.995, target=0.90, max_gain=1.5) -> torch.Tensor:
    vr, vi = vol_2thw[0], vol_2thw[1]
    mag = torch.sqrt(vr*vr + vi*vi)
    qq = torch.quantile(mag.reshape(-1), float(q))
    if not torch.isfinite(qq) or qq <= 1e-6:
        return vol_2thw
    gain_q = min(target / max(qq, 1e-6), max_gain)
    max_abs = vol_2thw.abs().amax()
    if max_abs > 0:
        gain_q = min(gain_q, float(1.0 / max_abs))
    return vol_2thw * gain_q

def scan_raw_shards(root: str, split: str):
    split_dir = os.path.join(root, split, "shards")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing {split_dir}")
    paths = [os.path.join(split_dir, fn) for fn in sorted(os.listdir(split_dir)) if fn.endswith(".h5")]
    if not paths:
        raise RuntimeError(f"No .h5 shards under {split_dir}")
    return paths

# ---- CINEDataset-matching spatial helpers ----

def _ensure_min_spatial(vol_2thw: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    C, T, H, W = vol_2thw.shape
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h == 0 and pad_w == 0:
        return vol_2thw
    x = vol_2thw.permute(1, 0, 2, 3).contiguous().view(1, T * C, H, W)
    ph_top = pad_h // 2; ph_bot = pad_h - ph_top
    pw_left = pad_w // 2; pw_right = pad_w - pw_left
    x = F.pad(x, (pw_left, pw_right, ph_top, ph_bot), mode="reflect")
    Hp, Wp = x.shape[-2], x.shape[-1]
    x = x.view(T, C, Hp, Wp).permute(1, 0, 2, 3).contiguous()
    return x

def _random_crop_hw(vol_2thw: torch.Tensor, ch: int, cw: int, rng: random.Random) -> torch.Tensor:
    vol_2thw = _ensure_min_spatial(vol_2thw, ch, cw)
    _, _, H, W = vol_2thw.shape
    y0 = 0 if H == ch else rng.randint(0, H - ch)
    x0 = 0 if W == cw else rng.randint(0, W - cw)
    return vol_2thw[:, :, y0:y0 + ch, x0:x0 + cw].contiguous()

def _pick_train_clip_indices(T: int, L: int, rng: random.Random) -> List[int]:
    if T >= L:
        start = rng.randint(0, T - L)
        return list(range(start, start + L))
    start = rng.randint(0, T - 1)
    return [(start + i) % T for i in range(L)]

def write_latent_item(hf: h5py.File, name: str, z: np.ndarray, attrs: Dict[str, Any]):
    g = hf.create_group(name)
    g.create_dataset("z", data=z, dtype="float32", compression="lzf", chunks=True)
    for k, v in attrs.items(): g.attrs[k] = v

# ---------------- emptiness logic ----------------

@torch.no_grad()
def _frames_nonempty_mask(
    clip_2thw: torch.Tensor,
    q: float,
    q_thr: float,
    px_thr: float,
    frac_thr: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    clip_2thw: [2, T(=7), H, W], on device
    A frame is 'non-empty' if (quantile >= q_thr) OR (frac_above_px_thr >= frac_thr).
    Returns:
      mask: [T] boolean
      stats: {'qvals': [T], 'frac': [T]}
    """
    mag = torch.sqrt(clip_2thw[0]**2 + clip_2thw[1]**2)        # [T,H,W]
    T = mag.shape[0]
    flat = mag.view(T, -1)

    qvals = torch.quantile(flat, float(q), dim=1)              # [T]
    frac  = (flat > float(px_thr)).float().mean(dim=1)         # [T]
    mask  = (qvals >= float(q_thr)) & (frac >= float(frac_thr))
    return mask, {"qvals": qvals, "frac": frac}

def _clip_is_acceptable(
    clip_2thw: torch.Tensor,
    empty_q: float,
    empty_q_thr: float,
    empty_px_thr: float,
    empty_frac_thr: float,
    min_nonempty: int,
    require_center_nonempty: bool,
) -> Tuple[bool, Dict[str, Any]]:
    mask, stats = _frames_nonempty_mask(clip_2thw, empty_q, empty_q_thr, empty_px_thr, empty_frac_thr)
    n_good = int(mask.sum().item())
    ok = (n_good >= int(min_nonempty))
    if require_center_nonempty:
        center_ok = bool(mask[int(clip_2thw.shape[1] // 2)].item())
        ok = ok and center_ok
    return ok, {"mask": mask, "n_good": n_good, **stats}

# ---------------- core ----------------

@torch.no_grad()
def precompute_train_latents_like_cine(
    in_root: str,
    out_root: str,
    vae_import: str,
    vae_class: str,
    vae_ckpt: str,
    strict_load: bool,
    fixed_L: int,
    crop_h: int,
    crop_w: int,
    samples_per_video: int,
    include_flip: bool,
    device: str,
    vae_batch: int,
    rng_seed: int,
    cfg: Dict[str, Any],
    # ---- new emptiness controls ----
    skip_empty: bool,
    empty_q: float,
    empty_q_thr: float,
    empty_px_thr: float,
    empty_frac_thr: float,
    min_nonempty: int,
    require_center_nonempty: bool,
):
    assert fixed_L == 7, "L must be 7 (fixed)."
    assert fixed_L % 2 == 1, "L must be odd."
    vae_batch = max(1, int(vae_batch))

    # ---- load frozen VAE
    VAE = dynamic_import(vae_import, vae_class)
    vae = VAE(**cfg["model"]["args"]).to(device).eval()
    if vae_ckpt and os.path.isfile(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        state = OrderedDict((k[10:] if k.startswith("_orig_mod.") else k, v) for k, v in state.items())
        vae.load_state_dict(state, strict=strict_load)
        print(f"[VAE] loaded {vae_ckpt}")
    for p in vae.parameters(): p.requires_grad_(False)

    # ---- latent geometry (nt very likely 4 for L=7)
    dummy = torch.zeros(2, fixed_L, crop_h, crop_w, device=device)
    enc = vae([dummy], op="encode")
    mu0 = enc[0][0] if isinstance(enc[0], (list, tuple)) else enc[0]
    if mu0.dim() == 5 and mu0.shape[0] == 1: mu0 = mu0.squeeze(0)
    Cz, nt, H_lat, W_lat = int(mu0.shape[0]), int(mu0.shape[1]), int(mu0.shape[2]), int(mu0.shape[3])
    print(f"[info] latent geometry: Cz={Cz} nt={nt} H'={H_lat} W'={W_lat} (L=7)")

    split = "train"
    in_shards = scan_raw_shards(in_root, split)

    # ---- outputs
    out_dir = os.path.join(out_root, split); ensure_dir(out_dir)
    out_shards = os.path.join(out_dir, "latent_shards"); ensure_dir(out_shards)
    meta_path = os.path.join(out_dir, "latents_meta.json")
    meta = {
        "split": split,
        "note": "training latents (μ) like CINEDataset(train): L=7, 80x80 random crop; may include horizontal flips",
        "latent_shape": {"Cz": Cz, "T_lat": nt, "H": H_lat, "W": W_lat},
        "pixel_clip": {"L": fixed_L, "crop_h": crop_h, "crop_w": crop_w},
        "include_flip": bool(include_flip),
        "emptiness": {
            "enabled": bool(skip_empty),
            "q": float(empty_q),
            "q_thr": float(empty_q_thr),
            "px_thr": float(empty_px_thr),
            "frac_thr": float(empty_frac_thr),
            "min_nonempty": int(min_nonempty),
            "require_center": bool(require_center_nonempty),
        },
        "items": []
    }

    SHARD_SIZE = 64
    item_counter, shard_idx = 0, 0
    hf, cur_path = None, None
    def roll():
        nonlocal hf, shard_idx, cur_path
        if hf is not None: hf.close()
        cur_path = os.path.join(out_shards, f"latent_{shard_idx:05d}.h5")
        hf = h5py.File(cur_path, "w", libver="latest")
        shard_idx += 1
    roll()

    rng = random.Random(rng_seed)

    buf_x: List[torch.Tensor] = []
    buf_meta: List[Dict[str, Any]] = []

    stats = {
        "attempted_clips": 0,
        "kept_clips": 0,
        "skipped_empty": 0,
    }

    def flush():
        nonlocal item_counter
        if not buf_x: return
        pairs = vae([x for x in buf_x], op="encode")
        mus = []
        for pr in pairs:
            mu_i = pr[0] if isinstance(pr, (list, tuple)) else pr
            if mu_i.dim() == 5 and mu_i.shape[0] == 1:
                mu_i = mu_i.squeeze(0)
            mus.append(mu_i.float().cpu().numpy())  # [Cz, nt, H', W']

        for meta_i, z_np in zip(buf_meta, mus):
            name = f"{item_counter:08d}"
            write_latent_item(hf, name, z_np, meta_i)
            meta["items"].append({
                "ds": os.path.basename(cur_path),
                "name": name,
                **{k: meta_i[k] for k in ("source_shard","source_key","sample_idx","flip")}
            })
            item_counter += 1
            if (item_counter % SHARD_SIZE) == 0:
                roll()

        buf_x.clear(); buf_meta.clear()

    print(f"[info] vae_batch={vae_batch}, shard_size={SHARD_SIZE}, samples_per_video={samples_per_video}, include_flip={include_flip}")
    if skip_empty:
        print(f"[info] skip_empty=True  q={empty_q}  q_thr={empty_q_thr}  px_thr={empty_px_thr}  "
              f"frac_thr={empty_frac_thr}  min_nonempty={min_nonempty}  require_center={require_center_nonempty}")

    for sp in in_shards:
        with h5py.File(sp, "r", libver="latest") as fr:
            keys = list(fr["volumes"].keys())
            for key in tqdm(keys, desc=f"train:{os.path.basename(sp)}", leave=False):
                vol = torch.from_numpy(fr["volumes"][key][()]).float().to(device)  # [2,T,H,W] in [-1,1]
                vol.clamp_(-1, 1)
                vol = robust_norm_qmag_global(vol)

                T = int(vol.shape[1])
                for si in range(int(samples_per_video)):
                    idxs = _pick_train_clip_indices(T, fixed_L, rng)
                    clip = vol[:, idxs]                               # [2,7,H,W]
                    clip = _random_crop_hw(clip, crop_h, crop_w, rng) # [2,7,80,80]

                    stats["attempted_clips"] += 1

                    if skip_empty:
                        ok, _dbg = _clip_is_acceptable(
                            clip, empty_q, empty_q_thr, empty_px_thr, empty_frac_thr,
                            min_nonempty=min_nonempty, require_center_nonempty=require_center_nonempty
                        )
                        if not ok:
                            stats["skipped_empty"] += 1
                            continue

                    # original
                    buf_x.append(clip)
                    buf_meta.append({
                        "source_shard": os.path.basename(sp),
                        "source_key": key,
                        "sample_idx": int(si),
                        "flip": 0,
                        "L_pixel": int(fixed_L),
                        "crop_h": int(crop_h),
                        "crop_w": int(crop_w),
                    })
                    stats["kept_clips"] += 1
                    if len(buf_x) >= vae_batch:
                        flush()

                    # optional horizontal flip (mirror W)
                    if include_flip:
                        clip_flip = clip.flip(-1)
                        # No need to re-check emptiness; flip preserves energy stats.
                        buf_x.append(clip_flip)
                        buf_meta.append({
                            "source_shard": os.path.basename(sp),
                            "source_key": key,
                            "sample_idx": int(si),
                            "flip": 1,
                            "L_pixel": int(fixed_L),
                            "crop_h": int(crop_h),
                            "crop_w": int(crop_w),
                        })
                        stats["kept_clips"] += 1
                        if len(buf_x) >= vae_batch:
                            flush()

    flush()
    if hf is not None: hf.close()
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    kept = stats["kept_clips"]
    skipped = stats["skipped_empty"]
    attempted = stats["attempted_clips"]
    print(f"[done] wrote {kept} latent clips  →  {meta_path}")
    print(f"[stats] attempted={attempted}  kept={kept}  skipped_empty={skipped}  "
          f"keep_rate={(kept/max(1,attempted)):.3f}")

def main():
    ap = argparse.ArgumentParser("Precompute training latents like CINEDataset(train), with optional horizontal flips.")
    ap.add_argument("--config", default="configs/vae.yaml")
    ap.add_argument("--in_root",  default="/storage/CINE_data")
    ap.add_argument("--out_root", default="/storage/CINE_data/latents_like_cine_no_empty_4")
    ap.add_argument("--vae_import", default="CardiacVAE.model.vae")
    ap.add_argument("--vae_class",  default="CardiacVAE")
    ap.add_argument("--vae_ckpt",   default="/storage/matt_models/cardiac_vae/videos/step_0195000/state.pt")
    ap.add_argument("--strict_load", action="store_true")

    ap.add_argument("--fixed_L", type=int, default=7)
    ap.add_argument("--crop_h",  type=int, default=80)
    ap.add_argument("--crop_w",  type=int, default=80)
    ap.add_argument("--samples_per_video", type=int, default=4)
    ap.add_argument("--include_flip", action="store_true")

    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--vae_batch", type=int, default=64)  # big is fine (no grads)
    ap.add_argument("--seed",      type=int, default=123)

    # ---- emptiness controls (match CINEDataset spirit) ----
    ap.add_argument("--skip_empty", action="store_true",
                    help="Drop clips where too many frames are near-empty.")
    ap.add_argument("--empty_q", type=float, default=0.99,
                    help="Quantile for per-frame energy scoring.")
    ap.add_argument("--empty_q_thr", type=float, default=0.3,
                    help="Threshold on that quantile to count a frame as non-empty.")
    ap.add_argument("--empty_px_thr", type=float, default=0.05,
                    help="Pixel floor used in the fraction-of-pixels test.")
    ap.add_argument("--empty_frac_thr", type=float, default=0.3,
                    help="Min fraction of pixels above empty_px_thr for a frame to be non-empty.")
    ap.add_argument("--min_nonempty", type=int, default=5,
                    help="Require at least this many non-empty frames in a 7-frame clip.")
    ap.add_argument("--require_center_nonempty", action="store_true",
                    help="If set, the center frame (t=3) must be non-empty.")

    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    precompute_train_latents_like_cine(
        in_root=args.in_root,
        out_root=args.out_root,
        vae_import=args.vae_import,
        vae_class=args.vae_class,
        vae_ckpt=args.vae_ckpt,
        strict_load=bool(args.strict_load),
        fixed_L=int(args.fixed_L),
        crop_h=int(args.crop_h),
        crop_w=int(args.crop_w),
        samples_per_video=int(args.samples_per_video),
        include_flip=bool(args.include_flip),
        device=args.device,
        vae_batch=int(args.vae_batch),
        rng_seed=int(args.seed),
        cfg=cfg,
        # pass emptiness args through
        skip_empty=bool(args.skip_empty),
        empty_q=float(args.empty_q),
        empty_q_thr=float(args.empty_q_thr),
        empty_px_thr=float(args.empty_px_thr),
        empty_frac_thr=float(args.empty_frac_thr),
        min_nonempty=int(args.min_nonempty),
        require_center_nonempty=bool(args.require_center_nonempty),
    )

if __name__ == "__main__":
    main()
