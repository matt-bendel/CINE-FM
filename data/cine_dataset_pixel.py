# data/cine_dataset_pixel.py
import os, mmap, pickle, random
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import scipy.stats as stats

def _complex_gamma_correction_flat(z_flat: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """Apply gamma to magnitudes >= 1, preserve phase. z_flat is complex1d."""
    phase = np.angle(z_flat)
    mag   = np.abs(z_flat)
    mask  = (mag >= 1.0)
    mag[mask] = mag[mask] ** gamma
    return mag * np.exp(1j * phase)

def preprocess_complex_2thw(x_2thw: torch.Tensor,
                            gamma: float = 0.1,
                            scale: float = 0.2,
                            final_div: float = 1.15) -> torch.Tensor:
    """
    Mirror colleague’s normalization:
      1) multiply by 0.2
      2) complex gamma correction (γ=0.1) on magnitudes >=1
      3) divide by fixed max ~1.15
    """
    device = x_2thw.device
    dtype  = x_2thw.dtype
    r = x_2thw[0].cpu().numpy()
    i = x_2thw[1].cpu().numpy()
    z = (r + 1j * i) * scale
    zf = z.reshape(-1)
    zf = _complex_gamma_correction_flat(zf, gamma=gamma)
    z  = zf.reshape(z.shape) / final_div
    out = np.stack([np.real(z), np.imag(z)], axis=0)  # [2,T,H,W]
    return torch.from_numpy(out).to(device=device, dtype=dtype)

class CINEPixelDataset(Dataset):
    """
    Loads a memory-mapped pickle list of arrays/tensors shaped [2, T_full, H_full, W_full]
    and yields fixed-size clips [2, 8, 64, 64] with colleague-matched preprocessing.
    """
    def __init__(self,
                 data_path: str,
                 t_frames: int = 8,
                 crop_hw: Tuple[int, int] = (64, 64),
                 std_scale: float = 1.5,
                 apply_preproc: bool = True,
                 log_stats: bool = True):
        assert os.path.isfile(data_path), f"File not found: {data_path}"
        with open(data_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            self.data_list = pickle.loads(mm)
            mm.close()

        self.t_frames = int(t_frames)
        self.t_H, self.t_W = int(crop_hw[0]), int(crop_hw[1])
        self.std_scale = float(std_scale)
        self.apply_preproc = bool(apply_preproc)

        if log_stats and len(self.data_list) > 0:
            # Peek stats (without loading everything to GPU)
            mx, mn = -1e9, 1e9
            sample_shape = None
            for i in range(min(len(self.data_list), 64)):
                arr = self.data_list[i]
                if not isinstance(arr, torch.Tensor):
                    arr = torch.tensor(arr, dtype=torch.float32)
                mx = max(mx, float(arr.max()))
                mn = min(mn, float(arr.min()))
                sample_shape = tuple(arr.shape)
            print(f"[CINEPixelDataset] loaded={len(self.data_list)} sample_shape={sample_shape} "
                  f"min={mn:.6f} max={mx:.6f}")
            print("[CINEPixelDataset] using truncated-Gaussian spatial crops + circular roll + flips")
            print(f"[CINEPixelDataset] output fixed shape: (2, {self.t_frames}, {self.t_H}, {self.t_W})")

    def __len__(self):
        return len(self.data_list)

    def _truncnorm_coords(self, H: int, W: int):
        # Match colleague’s sampling bias (center-favoring)
        top_mean = (H - self.t_H) / 2.0
        top_std  = (H - self.t_H) / 6.0 * (self.std_scale / 2.0)
        top_a, top_b = 0, H - self.t_H

        left_mean = (W - self.t_W) / 2.0
        left_std  = (W - self.t_W) / 6.0 * self.std_scale
        left_a, left_b = -(self.t_W - 1), W - 1  # allow negative for circular roll

        top = stats.truncnorm((top_a - top_mean) / (top_std + 1e-8),
                              (top_b - top_mean) / (top_std + 1e-8),
                              loc=top_mean, scale=top_std + 1e-8).rvs()
        left = stats.truncnorm((left_a - left_mean) / (left_std + 1e-8),
                               (left_b - left_mean) / (left_std + 1e-8),
                               loc=left_mean, scale=left_std + 1e-8).rvs()
        return int(top), int(left)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # x: [2, T_full, H_full, W_full]
        _, T, H, W = x.shape

        # 1) Temporal circular crop to exactly 8 frames
        start = random.randint(0, max(0, T - self.t_frames))
        idxs  = [(start + i) % T for i in range(self.t_frames)]
        x = x[:, idxs, :, :]  # [2, 8, H, W]

        # 2) Spatial: circular horizontal roll then crop (top, 0) with width 64
        top, left = self._truncnorm_coords(H, W)
        x = torch.roll(x, shifts=-left, dims=-1)          # circular shift along width
        x = TF.crop(x, top, 0, self.t_H, self.t_W)        # crop to (64,64)

        # 3) Random flips
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-2])

        # 4) Colleague-matched preprocessing (scale→gamma→fixed max)
        if self.apply_preproc:
            x = preprocess_complex_2thw(x, gamma=0.1, scale=0.2, final_div=1.15)

        # Final guard
        assert x.shape == (2, self.t_frames, self.t_H, self.t_W), f"Got {tuple(x.shape)}"
        return x

    def getshapes(self):
        return (len(self.data_list), 2, self.t_frames, self.t_H, self.t_W)
