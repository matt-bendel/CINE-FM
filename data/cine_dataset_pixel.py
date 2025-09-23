# data/cine_pixel_dataset.py
import os, json, h5py, torch, random
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class CINEPixelDataset(torch.utils.data.Dataset):
    """
    Pixel-space CINE MRI dataset (no VAE), single mode = videos.

    • Train:
        - returns [2, L, 64, 64]
          L defaults to 8 (even allowed). If fixed_train_L is None, a value
          is drawn from t_choices.

    • Val/Test:
        - returns full volume [2, T_odd, H, W] after GLOBAL q-mag normalization
          (drops last frame if T is even).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",              # "train" | "val" | "test"
        # temporal behavior (EVEN length allowed; default 8)
        fixed_train_L: Optional[int] = 8,
        t_choices: Optional[List[int]] = None,   # used only if fixed_train_L is None
        # spatial crop config (train only)
        crop_h: int = 64,
        crop_w: int = 64,
        # normalization
        normalize: str = "qmag",           # "qmag" or "none"
        norm_q: float = 0.995,
        norm_target: float = 0.98,
        norm_max_gain: float = None,
        # misc
        seed: int = 123,
        **kwargs,
    ):
        assert split in ("train", "val", "test")
        self.root  = root
        self.split = split

        # temporal selection
        self.fixed_train_L = fixed_train_L
        if (self.fixed_train_L is not None) and (self.fixed_train_L < 1):
            raise ValueError("fixed_train_L must be >= 1 if provided.")
        if t_choices is None:
            t_choices = [4, 6, 8, 10, 12]
        self.t_choices = [int(t) for t in t_choices if t >= 1]
        assert len(self.t_choices) > 0, "t_choices must include lengths >= 1"

        # crop sizes (train only)
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)

        # normalization
        self.normalize = str(normalize).lower()
        assert self.normalize in ("qmag", "unitmax", "percentile", "none")
        self.norm_q = float(norm_q)
        self.norm_target = float(norm_target)
        self.norm_max_gain = float(norm_max_gain)

        # files
        split_dir = os.path.join(root, split)
        shards_dir = os.path.join(split_dir, "shards")
        meta_path  = os.path.join(split_dir, "split_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"split_meta.json not found at {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self._samples: List[Tuple[str, str]] = []
        for shard in sorted(os.listdir(shards_dir)):
            if not shard.endswith(".h5"): continue
            sp = os.path.join(shards_dir, shard)
            with h5py.File(sp, "r", libver="latest") as hf:
                if "volumes" not in hf: continue
                for key in hf["volumes"].keys():
                    self._samples.append((sp, key))
        if not self._samples:
            raise RuntimeError(f"No samples under {shards_dir}")

        self.rng = random.Random(seed)
        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self):
        return len(self._samples)

    # ---------- I/O + normalization (GLOBAL per volume) ----------
    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest", swmr=True)
            self._h5_cache[path] = f
        return f

    def _load_volume(self, shard_path, group_name) -> torch.Tensor:
        hf = self._get_file(shard_path)
        vol = torch.from_numpy(hf["volumes"][group_name][()]).float()  # [2, T, H, W] in [-1, 1]
        vol.clamp_(-1, 1)
        if self.normalize == "qmag":
            vol = self._robust_norm_qmag_global(vol)
        elif self.normalize == "unitmax":
            vol = self._unitmax_global(vol)
        elif self.normalize == "percentile":
            vol = self._norm_per_volume_percentile(vol)
        return vol

    def _unitmax_global(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Per-volume gain so that max(|vol|) == 1.0.
        Never up-scales beyond 1 if data already in [-1, 1].
        """
        max_abs = vol.abs().amax()
        if not torch.isfinite(max_abs) or max_abs <= 0:
            return vol
        gain = min(1.0 / max_abs.item(), 1.0)  # don't boost above 1
        return (vol * gain).clamp_(-1.0, 1.0)

    def _norm_per_volume_percentile(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Per-volume complex gain based on a high percentile of magnitude.
        One scalar 'gain' applied to both channels; no shifts; then hard-clip to [-1,1].
        """
        # magnitude over the whole volume
        mag = torch.sqrt(vol[0].pow(2) + vol[1].pow(2))       # [T,H,W]
        flat = mag.reshape(-1)
        hi = torch.quantile(flat, self.norm_q)
        if not torch.isfinite(hi) or hi <= 1e-8:
            return vol

        # desired: hi -> norm_target
        gain = self.norm_target / float(hi)

        # optional safety cap (disabled by default)
        if self.norm_max_gain is not None:
            gain = min(gain, float(self.norm_max_gain))

        # also guarantee we never exceed [-1,1] due to outliers after scaling
        max_abs = vol.abs().amax()
        if max_abs > 0:
            gain_clip = 1.0 / float(max_abs)
            gain = min(gain, gain_clip)

        vol = vol * gain
        vol.clamp_(-1.0, 1.0)   # hard clamp
        return vol

    def _robust_norm_qmag_global(self, vol: torch.Tensor) -> torch.Tensor:
        vr, vi = vol[0], vol[1]
        mag = torch.sqrt(vr * vr + vi * vi)
        q = torch.quantile(mag.reshape(-1), self.norm_q)
        if not torch.isfinite(q) or q <= 1e-6:
            return vol
        gain_q = self.norm_target / max(q, 1e-6)
        gain_q = min(gain_q, self.norm_max_gain)
        # prevent clipping: ensure max(|vol|)*gain <= 1.0
        max_abs = vol.abs().amax()
        gain_clip = (1.0 / max_abs).item() if max_abs > 0 else gain_q
        gain = min(gain_q, gain_clip)
        return vol * gain

    # ---------- spatial crop helpers (64×64) ----------
    def _ensure_min_spatial(self, vol: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
        C, T, H, W = vol.shape
        pad_h = max(0, min_h - H)
        pad_w = max(0, min_w - W)
        if pad_h == 0 and pad_w == 0:
            return vol
        x = vol.permute(1, 0, 2, 3).contiguous().view(1, T * C, H, W)
        ph_top = pad_h // 2
        ph_bot = pad_h - ph_top
        pw_left = pad_w // 2
        pw_right = pad_w - pw_left
        x = F.pad(x, (pw_left, pw_right, ph_top, ph_bot), mode="reflect")
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.view(T, C, Hp, Wp).permute(1, 0, 2, 3).contiguous()
        return x

    def _random_crop_hw(self, vol: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
        vol = self._ensure_min_spatial(vol, ch, cw)
        _, _, H, W = vol.shape
        y0 = 0 if H == ch else self.rng.randint(0, H - ch)
        x0 = 0 if W == cw else self.rng.randint(0, W - cw)
        return vol[:, :, y0:y0 + ch, x0:x0 + cw].contiguous()

    # ---------- temporal helpers ----------
    @staticmethod
    def _oddify(vol: torch.Tensor) -> torch.Tensor:
        T = vol.shape[1]
        return vol if (T % 2 == 1) else vol[:, :T - 1]

    def _pick_train_clip_indices(self, T: int, L: int) -> List[int]:
        """Random contiguous window of length L; circular wrap if needed."""
        if T >= L:
            start = self.rng.randint(0, T - L)
            return list(range(start, start + L))
        start = self.rng.randint(0, T - 1)
        return [(start + i) % T for i in range(L)]

    # ---------- dataset API ----------
    def __getitem__(self, idx: int):
        shard_path, key = self._samples[idx]
        vol = self._load_volume(shard_path, key)  # [2, T, H, W]

        if self.split == "train":
            # choose L then take contiguous clip
            L = self.fixed_train_L if self.fixed_train_L is not None else self.rng.choice(self.t_choices)
            idxs = self._pick_train_clip_indices(vol.shape[1], L)
            clip = vol[:, idxs]  # [2, L, H, W]
            # spatial crop to 64×64
            clip = self._random_crop_hw(clip, self.crop_h, self.crop_w)
            return clip  # [2, L, 64, 64]
        else:
            # full video, oddified (no crop)
            return self._oddify(vol).contiguous()

    # ---------- cleanup ----------
    def close(self):
        for p, f in list(self._h5_cache.items()):
            try: f.close()
            except: pass
        self._h5_cache.clear()

    def __del__(self):
        try: self.close()
        except: pass
