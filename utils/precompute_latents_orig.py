#!/usr/bin/env python3
import os, sys, json, math, argparse, importlib, random
from typing import Dict, Any, List, Tuple, Optional
import h5py
import numpy as np
from tqdm.auto import tqdm
import torch
import yaml

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# ------------------- utils -------------------
def dynamic_import(import_path: str, class_name: str):
    return getattr(importlib.import_module(import_path), class_name)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------- IO helpers -------------------
def scan_shards_under_split(root: str, split: str) -> List[Tuple[str, List[str]]]:
    """
    Returns list of (h5_path, [keys]) for the split.
    Mirrors CINEDataset's discovery (…/split/shards/*.h5 under 'volumes').
    """
    split_dir = os.path.join(root, split, "shards")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing shards dir: {split_dir}")
    out = []
    for fn in sorted(os.listdir(split_dir)):
        if not fn.endswith(".h5"): continue
        p = os.path.join(split_dir, fn)
        with h5py.File(p, "r", libver="latest") as hf:
            if "volumes" not in hf: continue
            keys = list(hf["volumes"].keys())
        if keys:
            out.append((p, keys))
    if not out:
        raise RuntimeError(f"No volumes found under {split_dir}")
    return out

def write_latent_item(hf: h5py.File, name: str, z: np.ndarray, attrs: Dict[str, Any]):
    g = hf.create_group(name)
    g.create_dataset("z", data=z, dtype="float32", compression="lzf", chunks=True)
    for k, v in attrs.items():
        g.attrs[k] = v

# ------------------- val/test utilities -------------------
def oddify_T(x_2THW: torch.Tensor) -> torch.Tensor:
    T = x_2THW.shape[1]
    return x_2THW if (T % 2 == 1) else x_2THW[:, :T-1].contiguous()

def center_crop_hw(x_2THW: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
    _, _, H, W = x_2THW.shape
    y0 = max(0, (H - ch) // 2); x0 = max(0, (W - cw) // 2)
    y1 = min(H, y0 + ch);       x1 = min(W, x0 + cw)
    # if needed, reflect-pad to reach ch,cw
    pad_top = max(0, -y0); pad_left = max(0, -x0)
    pad_bot = max(0, ch - (y1 - y0) - pad_top)
    pad_right = max(0, cw - (x1 - x0) - pad_left)
    if pad_top or pad_bot or pad_left or pad_right:
        # reflect pad treating (2*T) as channels to keep real/imag together
        C, T, H0, W0 = x_2THW.shape
        x = x_2THW.permute(1,0,2,3).reshape(1, T*C, H0, W0)
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="reflect")
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.view(T, C, Hp, Wp).permute(1,0,2,3).contiguous()
        _, _, H, W = x.shape
        y0, x0 = 0, 0
        y1, x1 = ch, cw
        return x[:, :, y0:y1, x0:x1]
    return x_2THW[:, :, y0:y1, x0:x1].contiguous()

def sliding_temporal_starts(T_total: int, L: int, stride: int) -> List[int]:
    if T_total <= L: return [0]
    s = max(1, int(stride))
    starts = list(range(0, T_total - L + 1, s))
    if starts[-1] + L < T_total:
        starts.append(T_total - L)
    return starts

# ------------------- main precompute -------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Precompute VAE latents using the SAME sampling as CINEDataset(videos).")
    ap.add_argument("--config", default="configs/vae.yaml")
    ap.add_argument("--data_root", required=True, help="Root that contains train/val/test/shards")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--out_root", required=True, help="Where to write latents/<split>/")
    # VAE
    ap.add_argument("--vae_import", default="CardiacVAE.model.vae")
    ap.add_argument("--vae_class",  default="CardiacVAE")
    ap.add_argument("--vae_ckpt",   required=True)
    ap.add_argument("--strict_load", action="store_true")
    # Sampling to mimic CINEDataset
    ap.add_argument("--crop_h", type=int, default=80)
    ap.add_argument("--crop_w", type=int, default=80)
    ap.add_argument("--t_choices", type=int, nargs="*", default=[1,3,5,7,9,11],
                    help="Odd lengths used in training videos mode; ignored if fixed_train_L is set.")
    ap.add_argument("--fixed_train_L", type=int, default=7,
                    help="If set (odd), forces this L for train; also used for val/test windowing.")
    # Train-split: how many random clips per source volume
    ap.add_argument("--clips_per_volume", type=int, default=64,
                    help="Train: number of random (L, 80x80) clips drawn per volume (deterministic via seed).")
    # Val/Test: deterministic windowing
    ap.add_argument("--val_stride_t", type=int, default=6, help="Temporal stride for val/test sliding windows.")
    # batching / device / IO
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_chunks", type=int, default=128, help="Microbatch size for VAE.encode (list API).")
    ap.add_argument("--shard_size", type=int, default=4000, help="Max items per output HDF5 shard.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    assert args.fixed_train_L % 2 == 1, "fixed_train_L must be odd"
    set_all_seeds(args.seed)

    # Load VAE
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    VAE = dynamic_import(args.vae_import, args.vae_class)
    vae = VAE(**cfg["model"]["args"]).to(args.device).eval()
    ck = torch.load(args.vae_ckpt, map_location="cpu")
    state = ck.get("ema", ck.get("model", ck))
    state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    vae.load_state_dict(state, strict=args.strict_load)
    for p in vae.parameters(): p.requires_grad_(False)
    print(f"[VAE] loaded {args.vae_ckpt}")

    # infer latent shape (Cz, nt, H', W') for the chosen L
    L = int(args.fixed_train_L)
    dummy = torch.zeros(2, L, args.crop_h, args.crop_w, device=args.device)
    enc = vae([dummy], op="encode")
    mu = enc[0][0] if isinstance(enc[0], (list, tuple)) else enc[0]
    if mu.dim() == 5 and mu.shape[0] == 1: mu = mu.squeeze(0)
    Cz, nt, Hlat, Wlat = int(mu.shape[0]), int(mu.shape[1]), int(mu.shape[2]), int(mu.shape[3])

    # scan data
    sources = scan_shards_under_split(args.data_root, args.split)

    # outputs
    out_split = os.path.join(args.out_root, args.split)
    out_shards = os.path.join(out_split, "latent_shards")
    ensure_dir(out_shards)
    meta_path = os.path.join(out_split, "latents_meta.json")
    meta: Dict[str, Any] = {
        "split": args.split,
        "note": "Latents computed using CINEDataset-like normalization & sampling",
        "latent_shape": {"Cz": Cz, "nt": nt, "H": Hlat, "W": Wlat},
        "input_shape": {"L": L, "H": args.crop_h, "W": args.crop_w},
        "clips_per_volume": (args.clips_per_volume if args.split=="train" else None),
        "val_stride_t": (args.val_stride_t if args.split!="train" else None),
        "items": []
    }

    shard_size = int(args.shard_size)
    item_counter, shard_idx = 0, 0
    hf: Optional[h5py.File] = None
    cur_path: Optional[str] = None

    def roll():
        nonlocal hf, shard_idx, cur_path
        if hf is not None: hf.close()
        cur_path = os.path.join(out_shards, f"latent_{shard_idx:05d}.h5")
        hf = h5py.File(cur_path, "w", libver="latest")
        shard_idx += 1

    roll()

    # ---- main loop ----
    for h5_path, keys in sources:
        # open once for speed
        f = h5py.File(h5_path, "r", libver="latest")
        vols = f["volumes"]

        for key in tqdm(keys, desc=f"{args.split}:{os.path.basename(h5_path)}", leave=False):
            x = torch.from_numpy(vols[key][()]).float()  # [2,T,H,W] in [-1,1]
            x.clamp_(-1, 1)

            # ----- normalization: GLOBAL q-mag (exactly like CINEDataset) -----
            vr, vi = x[0], x[1]
            mag = torch.sqrt(vr*vr + vi*vi)
            q = torch.quantile(mag.reshape(-1), 0.995)
            if torch.isfinite(q) and (q > 1e-6):
                gain_q = 0.90 / float(q)
                gain_q = min(gain_q, 1.5)
                max_abs = float(x.abs().amax())
                if max_abs > 0:
                    gain_q = min(gain_q, 1.0 / max_abs)
                x = x * gain_q

            # ----- split logic -----
            if args.split == "train":
                # deterministic RNG per (file,key) so repeated runs reproduce samples
                rng = random.Random((hash(h5_path) ^ hash(key) ^ args.seed) & 0xfffffff)
                Ttot, H, W = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])

                # draw clips_per_volume samples
                want = int(args.clips_per_volume)
                t_choices = args.t_choices if args.fixed_train_L is None else [L]

                clips: List[torch.Tensor] = []
                attrs: List[Dict[str,Any]] = []

                for _ in range(want):
                    L_use = L if args.fixed_train_L is not None else rng.choice([t for t in t_choices if t%2==1])
                    # temporal indices like CINEDataset._pick_train_clip_indices (with wrap)
                    if Ttot >= L_use:
                        start = rng.randint(0, Ttot - L_use)
                        idxs = list(range(start, start + L_use))
                    else:
                        start = rng.randint(0, Ttot - 1)
                        idxs = [ (start + i) % Ttot for i in range(L_use) ]
                    clip = x[:, idxs]  # [2,L_use,H,W]

                    # random 80x80 crop like CINEDataset._random_crop_hw
                    ch, cw = args.crop_h, args.crop_w
                    # reflect-pad to ensure >= (ch,cw)
                    pad_h = max(0, ch - H); pad_w = max(0, cw - W)
                    if pad_h or pad_w:
                        X = clip.permute(1,0,2,3).reshape(1, L_use*2, H, W)
                        ph_top = pad_h // 2; ph_bot = pad_h - ph_top
                        pw_left = pad_w // 2; pw_right = pad_w - pw_left
                        X = torch.nn.functional.pad(X, (pw_left, pw_right, ph_top, ph_bot), mode="reflect")
                        Hp, Wp = X.shape[-2], X.shape[-1]
                        clip = X.view(L_use, 2, Hp, Wp).permute(1,0,2,3).contiguous()
                        H, W = Hp, Wp
                    y0 = 0 if H == ch else rng.randint(0, H - ch)
                    x0 = 0 if W == cw else rng.randint(0, W - cw)
                    clip = clip[:, :, y0:y0+ch, x0:x0+cw].contiguous()  # [2,L_use,80,80]

                    # if L_use != L (rare if you force fixed), center-trim/pad to L
                    if L_use != L:
                        if L_use > L:
                            s = (L_use - L)//2
                            clip = clip[:, s:s+L]
                        else:
                            # mirror-pad in time to length L
                            need = L - L_use
                            left = need//2; right = need - left
                            clip_t = torch.cat([clip[:, :1].repeat(1,left,1,1), clip, clip[:, -1:].repeat(1,right,1,1)], dim=1)
                            clip = clip_t[:, :L]
                    assert clip.shape[1] == L and clip.shape[2] == args.crop_h and clip.shape[3] == args.crop_w

                    clips.append(clip.to(args.device))
                    attrs.append({
                        "src_shard": os.path.basename(h5_path),
                        "src_key": key,
                        "L": int(L),
                        "crop_yx": (int(y0), int(x0)),
                    })

                # encode in microbatches
                zs: List[torch.Tensor] = []
                for i in range(0, len(clips), args.batch_chunks):
                    batch_list = [c for c in clips[i:i+args.batch_chunks]]  # list of [2,L,80,80]
                    mus_logs = vae(batch_list, op="encode")                 # list of (mu, logv)
                    for pr in mus_logs:
                        mu = pr[0] if isinstance(pr, (list, tuple)) else pr
                        if mu.dim() == 5 and mu.shape[0] == 1:
                            mu = mu.squeeze(0)
                        zs.append(mu.detach().cpu())                        # [Cz,nt,H',W']

                # write
                for z, at in zip(zs, attrs):
                    name = f"{item_counter:08d}"
                    write_latent_item(hf, name, z.numpy(), at)
                    meta["items"].append({
                        "ds": os.path.basename(cur_path), "name": name, **at
                    })
                    item_counter += 1
                    if (item_counter % shard_size) == 0:
                        roll()

            else:
                # val/test: deterministic
                clip = oddify_T(x)                                          # [2,T_odd,H,W]
                clip = center_crop_hw(clip, args.crop_h, args.crop_w)       # [2,T_odd,80,80]
                Todd = int(clip.shape[1])
                starts = sliding_temporal_starts(Todd, L, args.val_stride_t)

                # build windows
                windows = [clip[:, s:s+L].contiguous().to(args.device) for s in starts]
                attrs = [{
                    "src_shard": os.path.basename(h5_path),
                    "src_key": key,
                    "L": int(L),
                    "t0": int(s),
                    "crop": "center",
                } for s in starts]

                # encode
                zs: List[torch.Tensor] = []
                for i in range(0, len(windows), args.batch_chunks):
                    batch_list = [w for w in windows[i:i+args.batch_chunks]]
                    mus_logs = vae(batch_list, op="encode")
                    for pr in mus_logs:
                        mu = pr[0] if isinstance(pr, (list, tuple)) else pr
                        if mu.dim() == 5 and mu.shape[0] == 1:
                            mu = mu.squeeze(0)
                        zs.append(mu.detach().cpu())

                for z, at in zip(zs, attrs):
                    name = f"{item_counter:08d}"
                    write_latent_item(hf, name, z.numpy(), at)
                    meta["items"].append({
                        "ds": os.path.basename(cur_path), "name": name, **at
                    })
                    item_counter += 1
                    if (item_counter % shard_size) == 0:
                        roll()

        f.close()

    if hf is not None: hf.close()
    ensure_dir(os.path.dirname(meta_path))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote {item_counter} latent items → {meta_path}")

if __name__ == "__main__":
    main()
