#!/usr/bin/env python3
import os, sys, glob, argparse, math, h5py, numpy as np, torch
import yaml

LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

# ---------- tiny helpers ----------
def dynamic_import(import_path: str, class_name: str):
    mod = __import__(import_path, fromlist=[class_name])
    return getattr(mod, class_name)

def first_latent_file(latent_root: str, split: str) -> str:
    d = os.path.join(latent_root, split, "latent_shards")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing directory: {d}")
    cands = sorted(glob.glob(os.path.join(d, "latent_*.h5")))
    if not cands:
        raise FileNotFoundError(f"No latent_*.h5 under {d}")
    return cands[0]

def first_item(path: str):
    with h5py.File(path, "r", libver="latest") as f:
        names = sorted(list(f.keys()))
        if not names:
            raise RuntimeError(f"{path} is empty")
        g = f[names[0]]
        z = np.array(g["z"][...], dtype=np.float32)  # [Cz, 7, P, H', W']
        A = g.attrs

        # required attrs written by precompute (patch/stride/P/temporal start)
        ph = int(A.get("ph")); pw = int(A.get("pw"))
        sh = int(A.get("sh")); sw = int(A.get("sw"))
        P  = int(A.get("P"))
        t0 = int(A.get("t0"))

        # NEW (see patch below). If absent, infer conservative values.
        n1 = int(A.get("n1", 0)); n2 = int(A.get("n2", 0))
        H  = int(A.get("H",  0)); W  = int(A.get("W",  0))

        if n1 * n2 != P:
            # infer a grid (row-major) from P if not provided
            if n1 == 0 or n2 == 0:
                n1 = int(math.floor(math.sqrt(P)))
                while n1 > 1 and P % n1 != 0:
                    n1 -= 1
                n2 = P // n1
            else:
                # fall back to factoring if given values are inconsistent
                n1 = int(math.floor(math.sqrt(P)))
                while n1 > 1 and P % n1 != 0:
                    n1 -= 1
                n2 = P // n1

        if H <= 0 or W <= 0:
            # minimal coverage size given grid/stride/patch
            H = (n1 - 1) * sh + ph
            W = (n2 - 1) * sw + pw

        return z, dict(ph=ph, pw=pw, sh=sh, sw=sw, P=P, n1=n1, n2=n2, H=H, W=W, t0=t0)

def axis_weights(L_eff: int, idx: int, n: int, O: int, device: torch.device):
    """Ramp weights so boundaries keep weight=1 and overlaps blend linearly."""
    has_prev = idx > 0
    has_next = idx < n - 1
    L_left  = min(O if has_prev else 0, L_eff)
    L_right = min(O if has_next else 0, L_eff)

    # if ramps would exceed the valid extent, re-split
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

def depatchify2d_rowmajor(
    patches_P_h_w: torch.Tensor,  # [P, ph, pw]
    n1: int, n2: int,
    ph: int, pw: int, sh: int, sw: int,
    H: int,  W: int,
) -> torch.Tensor:
    """Mirror your patchify2d_fixed_stride loop (j rows then k cols; P = n1*n2)."""
    device = patches_P_h_w.device
    dtype  = patches_P_h_w.dtype

    out_num = torch.zeros((H, W), dtype=dtype, device=device)
    out_den = torch.zeros((H, W), dtype=torch.float32, device=device)

    Oh = max(0, ph - sh)
    Ow = max(0, pw - sw)

    p = 0
    for j in range(n1):
        y0 = j * sh; y1 = min(y0 + ph, H); s1 = y1 - y0
        w1 = axis_weights(s1, j, n1, Oh, device)
        for k in range(n2):
            x0 = k * sw; x1 = min(x0 + pw, W); s2 = x1 - x0
            w2 = axis_weights(s2, k, n2, Ow, device)

            patch = patches_P_h_w[p][:s1, :s2]
            w = (w1[:, None] * w2[None, :])
            out_num[y0:y1, x0:x1] += (patch * w).to(out_num.dtype)
            out_den[y0:y1, x0:x1] += w
            p += 1

    out_den = torch.where(out_den == 0, torch.ones_like(out_den), out_den)
    return out_num / out_den.to(out_num.dtype)

def to_uint8(frame_1hw: torch.Tensor, lo=0.01, hi=0.99):
    f = torch.nan_to_num(frame_1hw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        return np.zeros((f.shape[-2], f.shape[-1]), dtype=np.uint8)
    lo_q = torch.quantile(flat, lo); hi_q = torch.quantile(flat, hi)
    g = torch.zeros_like(f) if (hi_q - lo_q) < 1e-8 else (f - lo_q) / (hi_q - lo_q)
    return (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0).numpy()

def save_video_uint8(frames_T_HW, path, fps=7):
    try:
        import imageio.v3 as iio
        iio.imwrite(path, frames_T_HW, fps=fps, codec="libx264", quality=8)
        print(f"[ok] saved {path}")
    except Exception as e:
        print(f"[warn] imageio/ffmpeg not available ({e}); writing PNGs.")
        base = os.path.splitext(path)[0]
        os.makedirs(base, exist_ok=True)
        from imageio.v3 import imwrite
        for i, fr in enumerate(frames_T_HW):
            imwrite(os.path.join(base, f"frame_{i:03d}.png"), fr)

# ---------- main ----------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Verify a precomputed latent window by decoding and depatchifying.")
    ap.add_argument("--latent_root", default="/storage/CINE_data/latents", help="Root used by precompute (…/<split>/latent_shards/latent_*.h5)")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--vae_import", default="CardiacVAE.model.vae")
    ap.add_argument("--vae_class",  default="CardiacVAE")
    ap.add_argument("--vae_ckpt",   default="/storage/matt_models/cardiac_vae/videos/step_0195000/state.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="latent_check.mp4")
    ap.add_argument("--config", default="configs/vae.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # pick first latent shard & item
    fpath = first_latent_file(args.latent_root, args.split)
    z_np, A = first_item(fpath)  # z: [Cz, 7, P, H', W']
    Cz, T, P, Hl, Wl = z_np.shape
    ph, pw, sh, sw, n1, n2, H, W = A["ph"], A["pw"], A["sh"], A["sw"], A["n1"], A["n2"], A["H"], A["W"]
    assert T == 4, f"Expected 4 frames per window; got {T}"

    # load VAE for decode
    VAE = dynamic_import(args.vae_import, args.vae_class)
    vae = VAE(**cfg["model"]["args"])
    ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    state = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items() }
    vae.load_state_dict(state, strict=False)
    vae = vae.to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    # decode per-patch latents -> [P, 2, 7, ph, pw]
    z = torch.from_numpy(z_np).to(device=device, dtype=torch.float32)
    dec = []
    CHUNK = 32
    for p0 in range(0, P, CHUNK):
        z_list = [ z[:, :, p].unsqueeze(0) for p in range(p0, min(p0+CHUNK, P)) ]  # list [1,Cz,7,H',W']
        outs = vae(z_list, op="decode")  # list of [1,2,7,ph,pw]
        dec += [o.squeeze(0) for o in outs]
    dec = torch.stack(dec, dim=0)  # [P, 2, 7, ph, pw]

    # depatchify spatially per time slice
    xr = torch.zeros((T, H, W), device=device, dtype=torch.float32)
    xi = torch.zeros_like(xr)
    for t in range(T):
        rP = dec[:, 0, t]  # [P, ph, pw]
        iP = dec[:, 1, t]
        xr[t] = depatchify2d_rowmajor(rP, n1, n2, ph, pw, sh, sw, H, W)
        xi[t] = depatchify2d_rowmajor(iP, n1, n2, ph, pw, sh, sw, H, W)

    # magnitude video
    mag = torch.sqrt(torch.clamp(xr * xr + xi * xi, min=0.0)).unsqueeze(0)  # [1,T,H,W]
    frames = np.stack([to_uint8(mag[:, t]) for t in range(T)], axis=0)      # [T,H,W]
    save_video_uint8(frames, args.out, fps=7)

if __name__ == "__main__":
    main()
