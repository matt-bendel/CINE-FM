# data/deg.py
#!/usr/bin/env python3
import os, sys, math, argparse, json
from typing import Tuple, Dict, Any, List

# Make project imports work when run as a module/script
LAUNCH_ROOT = os.path.abspath(os.getcwd())
if LAUNCH_ROOT not in sys.path:
    sys.path.insert(0, LAUNCH_ROOT)

import numpy as np
import torch

# ------------------- GRO sampling + masking -------------------

class MRIDeg:
    """
    GRO sampling + retrospective downsampling for CINE.
    - Build ky×t mask once via GRO, cache it.
    - Apply to full-frame videos [2,T,H,W] (real/imag).
    """
    def __init__(self, pe: int, fr: int, R: float, dsp: int = 0, verbose: bool = False):
        """
        pe: number of phase-encode lines (H)
        fr: number of frames (T)
        R:  acceleration factor (float or int)
        dsp: if 1, show matplotlib visualizations (requires MPL)
        """
        self.params = {"PE": int(pe), "FR": int(fr), "R": float(R), "dsp": int(dsp)}
        if verbose:
            print(f"[MRIDeg] GRO params: {self.params}")
        _, _, samp_e = MRIDeg._gro_fun(self.params)  # samp_e: [PE, FR, E]
        if samp_e.ndim == 3 and samp_e.shape[-1] == 1:
            samp_e = samp_e[..., 0]
        self.mask_ky_t = samp_e.astype(bool)  # [H(PE), T(FR)]

        # report effective acceleration from chosen n
        n = int(np.ceil(pe / R))
        self.eff_acc = pe / max(1, n)
        if verbose:
            print(f"[MRIDeg] effective acceleration ≈ {self.eff_acc:.2f} (target R={R})")

    def mask_tensor_THW(self, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """
        Build broadcastable [T,H,W] mask tensor (1.0 sampled, 0.0 otherwise).
        Readout (kx) is assumed fully sampled ⇒ tile along W.
        """
        H, T = int(self.mask_ky_t.shape[0]), int(self.mask_ky_t.shape[1])
        m_TH = torch.from_numpy(self.mask_ky_t.T.copy())  # [T,H]
        return m_TH[:, :, None].expand(T, H, int(width)).to(device=device, dtype=dtype)

    @staticmethod
    def _to_complex(x_2thw: torch.Tensor) -> torch.Tensor:
        """[2,T,H,W] → [T,H,W] complex."""
        return torch.complex(x_2thw[0], x_2thw[1])

    @staticmethod
    def _from_complex(x_thw: torch.Tensor) -> torch.Tensor:
        """[T,H,W] complex → [2,T,H,W] real/imag."""
        return torch.stack([x_thw.real, x_thw.imag], dim=0)

    def apply_to_video(self, x_2thw: torch.Tensor, fft_norm: str = "ortho", return_zero_filled: bool = True):
        assert x_2thw.dim() == 4 and x_2thw.shape[0] == 2, f"expected [2,T,H,W], got {tuple(x_2thw.shape)}"
        device = x_2thw.device
        _, T, H, W = x_2thw.shape
        if H != self.params["PE"] or T != self.params["FR"]:
            raise ValueError(f"[MRIDeg] video dims (T={T}, H={H}) must match mask (FR={self.params['FR']}, PE={self.params['PE']}).")

        # image -> k-space
        xc = MRIDeg._to_complex(x_2thw)              # [T,H,W] complex
        k  = torch.fft.fft2(xc, norm=fft_norm)       # uncentered FFT (DC at [0,0])

        # SHIFT to centered k-space (DC at center) to match GRO indexing
        k_c = torch.fft.fftshift(k, dim=(-2, -1))    # shift both ky (H) and kx (W)

        # build mask [T,H,W] (centered indexing along H)
        m = self.mask_tensor_THW(W, device=device, dtype=k.real.dtype)  # [T,H,W]
        k_masked_c = k_c * m

        # SHIFT back to uncentered before iFFT
        k_masked = torch.fft.ifftshift(k_masked_c, dim=(-2, -1))

        k_masked_ri = MRIDeg._from_complex(k_masked)  # [2,T,H,W]

        if not return_zero_filled:
            return k_masked_ri

        x_zf = torch.fft.ifft2(k_masked, norm=fft_norm)
        x_zf_ri = MRIDeg._from_complex(x_zf)
        return k_masked_ri, x_zf_ri

    # ---- numpy port of the GRO generator ----
    @staticmethod
    def _gro_fun(param: dict):
        n = int(np.ceil(param['PE'] / param['R']))
        FR = int(param['FR'])
        N  = int(param['PE'])
        E  = 1
        tau = 1
        PF  = 0
        s   = 2.2
        a   = 3
        dsp = int(param.get('dsp', 0))

        gr = (1 + np.sqrt(5)) / 2
        gr = 1 / (gr + tau - 1)

        Ns = int(np.ceil(N * 1 / s))
        k  = (N / 2 - Ns / 2) / ((Ns / 2) ** a)

        samp  = np.zeros((N, FR, E), dtype=float)
        PEInd = np.zeros(((n - PF) * FR, E), dtype=int)
        FRInd = np.zeros(((n - PF) * FR, 1), dtype=int)
        v0    = np.linspace(0.5, Ns + 0.5, n + PF, endpoint=False)

        for e in range(E):
            v0_e = v0 + 1 / E * Ns / (n + PF)
            kk = E - e
            for j in range(FR):
                v = (v0_e + (j * Ns / (n + PF)) * gr - 1) % Ns + 1
                v = np.where(v >= (Ns + 0.5), v - Ns, v)

                if N % 2 == 0:
                    vC = v - k * np.sign((Ns / 2 + 0.5) - v) * (np.abs((Ns / 2 + 0.5) - v)) ** a + (N - Ns) / 2 + 0.5
                    vC = np.where(vC >= (N + 0.5), vC - N, vC)
                else:
                    vC = v - k * np.sign((Ns / 2 + 0.5) - v) * (np.abs((Ns / 2 + 0.5) - v)) ** a + (N - Ns) / 2

                vC = np.round(np.sort(vC)).astype(int)
                vC = vC[PF:] - 1  # 0-based ky indices

                if j % 2 == 0:
                    PEInd[j * n : (j + 1) * n, e] = vC
                else:
                    PEInd[j * n : (j + 1) * n, e] = vC[::-1]

                FRInd[j * n : (j + 1) * n] = j
                samp[vC, j, e] = samp[vC, j, e] + kk

        if dsp == 1:
            try:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                im0 = axs[0].imshow(samp[:, :, 0], cmap='gray')
                axs[0].set_xlabel('t'); axs[0].set_ylabel(r'$k_y$')
                fig.colorbar(im0, ax=axs[0])
                len_ind = min(len(PEInd), 120)
                axs[1].plot(np.arange(len_ind), PEInd[:len_ind, 0], '.-')
                axs[1].set_xlabel('Acq order'); axs[1].set_ylabel(r'$k_y$')
                plt.show()
            except Exception as e:
                print(f"[MRIDeg] dsp plot failed: {e}")

        return PEInd, FRInd, samp.astype(bool)

# ------------------- Small IO / viz helpers -------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

@torch.no_grad()
def _complex_mag(x_2thw: torch.Tensor) -> torch.Tensor:
    # [2,T,H,W] -> [1,T,H,W]
    return (x_2thw.pow(2).sum(dim=0, keepdim=True)).sqrt()

@torch.no_grad()
def _video_to_uint8(frames_1thw: torch.Tensor, q_lo: float = 1.0, q_hi: float = 99.0) -> np.ndarray:
    """Percentile scaling over the whole video to reduce flicker. Returns [T,H,W] uint8."""
    f = torch.nan_to_num(frames_1thw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = f.flatten()
    if flat.numel() == 0:
        T, _, H, W = frames_1thw.shape
        return np.zeros((T, H, W), dtype=np.uint8)
    lo = torch.quantile(flat, q_lo / 100.0)
    hi = torch.quantile(flat, q_hi / 100.0)
    g = torch.zeros_like(f) if (hi - lo) < 1e-8 else (f - lo) / (hi - lo)
    g = (g.clamp_(0, 1) * 255.0).round().to(torch.uint8).squeeze(0)  # [T,H,W]
    return g.numpy()

def _save_video_uint8(arr_thw_u8: np.ndarray, path: str, fps: int = 7):
    import imageio.v2 as iio
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".gif"): path = path + ".mp4"
    try:
        iio.mimsave(path, list(arr_thw_u8), fps=int(fps))
    except Exception:
        # fallback to gif
        iio.mimsave(os.path.splitext(path)[0] + ".gif", list(arr_thw_u8), fps=int(fps))
    return path

def _save_mask_png(mask_ky_t: np.ndarray, path: str):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mask_ky_t.astype(float), cmap='gray', origin='upper', aspect='auto')
        ax.set_xlabel('t'); ax.set_ylabel(r'$k_y$'); fig.colorbar(im, ax=ax)
        fig.savefig(path, bbox_inches="tight", dpi=160); plt.close(fig)
    except Exception:
        # minimal fallback (no colorbar, no axes)
        import imageio.v2 as iio
        img = (mask_ky_t.astype(float) * 255.0).astype(np.uint8)
        iio.imwrite(path, img)

# ------------------- Dataset loader (validation) -------------------

def _dynamic_import(import_path: str, class_name: str):
    mod = __import__(import_path, fromlist=[class_name])
    return getattr(mod, class_name)

def _load_one_validation_video(cfg: Dict[str, Any], split: str = "val", device=None) -> torch.Tensor:
    """
    Returns one video tensor [2,T,H,W] (float32) from CINEDataset-style config.
    split: 'val' or 'test'
    """
    from torch.utils.data import DataLoader

    ds_key = "val_dataset" if split == "val" else "test_dataset"
    if ds_key not in cfg:
        # fallback to other key names used elsewhere
        ds_key = "val_dataset" if "val_dataset" in cfg else "test_dataset"
    ds_cfg = cfg[ds_key]
    name = ds_cfg["name"]
    if name != "CINEDataset":
        raise ValueError(f"Expected CINEDataset; got {name}")

    DS = _dynamic_import("data.cine_dataset", "CINEDataset")
    dataset = DS(**ds_cfg.get("args", {}))
    dl_cfg = cfg.get("dataloader", {})
    bs = int(dl_cfg.get("val_batch_size", 1))
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=dl_cfg.get("num_workers", 2),
                        pin_memory=dl_cfg.get("pin_memory", True), collate_fn=lambda x: x, drop_last=False)

    # ragged collate ⇒ each batch item is a list of [2,T,H,W] tensors
    for batch in loader:
        for x in batch:
            x = x.to(device=device, dtype=torch.float32)
            if x.dim() == 3:  # [2,H,W] → treat as single-frame video
                x = x.unsqueeze(1)
            assert x.dim() == 4 and x.shape[0] == 2, f"Bad sample shape {tuple(x.shape)}"
            return x
    raise RuntimeError("No validation samples found.")

# ------------------- Main test routine -------------------

@torch.no_grad()
def run_test(config_path: str, split: str, sample_idx: int, R: float, out_dir: str, fps: int = 7, cpu: bool = False):
    with open(config_path, "r") as f:
        import yaml
        cfg = yaml.safe_load(f)

    device = torch.device("cpu" if (cpu or not torch.cuda.is_available()) else "cuda")
    torch.backends.cudnn.benchmark = True

    # grab one video (we iterate until sample_idx)
    vid = None
    if sample_idx == 0:
        vid = _load_one_validation_video(cfg, split=split, device=device)
    else:
        # slower path: iterate to the desired index
        from torch.utils.data import DataLoader
        ds_key = "val_dataset" if split == "val" else "test_dataset"
        DS = _dynamic_import("data.cine_dataset", "CINEDataset")
        dataset = DS(**cfg[ds_key].get("args", {}))
        dl_cfg = cfg.get("dataloader", {})
        loader = DataLoader(dataset, batch_size=dl_cfg.get("val_batch_size", 1), shuffle=False,
                            num_workers=dl_cfg.get("num_workers", 2),
                            pin_memory=dl_cfg.get("pin_memory", True), collate_fn=lambda x: x, drop_last=False)
        count = 0
        for batch in loader:
            for x in batch:
                if count == sample_idx:
                    vid = x.to(device=device, dtype=torch.float32)
                    if vid.dim() == 3: vid = vid.unsqueeze(1)
                    break
                count += 1
            if vid is not None: break
        if vid is None:
            raise IndexError(f"sample-idx {sample_idx} out of range.")

    _, T, H, W = vid.shape
    print(f"[deg] sample: T={T}, H={H}, W={W}")

    # build mask + apply
    deg = MRIDeg(pe=H, fr=T, R=R, dsp=0, verbose=True)
    k_masked, x_zf = deg.apply_to_video(vid, fft_norm="ortho", return_zero_filled=True)

    # outputs
    out_dir = _ensure_dir(out_dir)
    # 1) save mask visualization
    _save_mask_png(deg.mask_ky_t, os.path.join(out_dir, "mask_ky_t.png"))

    # 2) save zero-filled recon magnitude video
    zf_mag = _complex_mag(x_zf)  # [1,T,H,W]
    zf_u8 = _video_to_uint8(zf_mag, q_lo=1.0, q_hi=99.0)  # [T,H,W] uint8
    _save_video_uint8(zf_u8, os.path.join(out_dir, "zfr_mag.mp4"), fps=fps)

    # 3) save k-space log-magnitude video (masked)
    kmag = _complex_mag(k_masked)           # [1,T,H,W] magnitude
    klog = torch.log1p(kmag / (kmag.quantile(0.95) + 1e-8))  # gentle compression
    k_u8  = _video_to_uint8(klog, q_lo=1.0, q_hi=99.5)
    _save_video_uint8(k_u8, os.path.join(out_dir, "kspace_logmag.mp4"), fps=fps)

    # 4) (optional) save a tiny JSON with meta
    meta = {
        "T": int(T), "H": int(H), "W": int(W),
        "R_target": float(R), "R_effective": float(deg.eff_acc),
        "paths": {
            "mask_png": "mask_ky_t.png",
            "zfr_video": "zfr_mag.mp4",
            "kspace_video": "kspace_logmag.mp4",
        }
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[deg] saved outputs in: {out_dir}")

# ------------------- CLI -------------------

def main():
    ap = argparse.ArgumentParser(description="GRO mask test: mask a validation CINE video and save ZFR/k-space/mask.")
    ap.add_argument("--config", type=str, default="configs/vae.yaml", help="YAML with val/test dataset args (CINEDataset).")
    ap.add_argument("--split",  type=str, default="val", choices=["val","test"], help="Which split to pull the sample from.")
    ap.add_argument("--sample-idx", type=int, default=0, help="Index of sample in the chosen split.")
    ap.add_argument("--R", type=float, default=6.0, help="Acceleration factor.")
    ap.add_argument("--out", type=str, default="deg_out", help="Output directory.")
    ap.add_argument("--fps", type=int, default=7, help="Video FPS for saved mp4/gif.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = ap.parse_args()
    run_test(args.config, args.split, args.sample_idx, args.R, args.out, fps=args.fps, cpu=args.cpu)

if __name__ == "__main__":
    main()
