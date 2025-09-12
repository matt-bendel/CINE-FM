# data/cine_dataset.py
import os, json, h5py, torch, random
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class CINEDataset(torch.utils.data.Dataset):
    """
    VAE dataset for complex CINE MRI.

    Train:
      - stage_mode="pretrain_2d": returns [2, 1, 80, 80] (single random frame, random 80x80 crop)
      - stage_mode="videos":      returns [2, L, 80, 80] (L odd; contiguous window with wrap if needed, random 80x80 crop)

    Val/Test:
      - returns full volume [2, T_odd, H, W] (no spatial crop), dropping last frame if T even

    The ONLY change from the prior behavior is:
      - normalization is now GLOBAL per volume (q-mag) BEFORE any temporal slicing or spatial cropping.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        stage_mode: str = "videos",      # "pretrain_2d" or "videos"
        # temporal behavior
        t_choices: Optional[List[int]] = None,   # odd lengths to sample in training (videos mode)
        fixed_train_L: Optional[int] = None,     # if set (odd), overrides t_choices in training
        # spatial crop config
        crop_h: int = 80,
        crop_w: int = 80,
        # normalization
        normalize: str = "qmag",         # "qmag" or "none"
        norm_q: float = 0.995,
        norm_target: float = 0.90,
        norm_max_gain: float = 1.5,
        # misc
        seed: int = 123,
        **kwargs,                        # ignore extra keys
    ):
        assert split in ("train", "val", "test")
        assert stage_mode in ("pretrain_2d", "videos")
        self.root = root
        self.split = split
        self.stage_mode = stage_mode

        # temporal selection
        if t_choices is None:
            t_choices = [1, 3, 5, 7, 9, 11]
        self.t_choices = [int(t) for t in t_choices if (t % 2 == 1 and t >= 1)]
        assert len(self.t_choices) > 0, "t_choices must include odd lengths >=1"

        if fixed_train_L is not None:
            assert fixed_train_L % 2 == 1 and fixed_train_L >= 1, "fixed_train_L must be odd and >=1"
        self.fixed_train_L = fixed_train_L

        # crop sizes
        self.crop_h = int(crop_h)
        self.crop_w = int(crop_w)

        # normalization
        self.normalize = str(normalize).lower()
        assert self.normalize in ("qmag", "none")
        self.norm_q = float(norm_q)
        self.norm_target = float(norm_target)
        self.norm_max_gain = float(norm_max_gain)

        # files
        split_dir = os.path.join(root, split)
        shards_dir = os.path.join(split_dir, "shards")
        meta_path = os.path.join(split_dir, "split_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"split_meta.json not found at {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self._samples: List[Tuple[str, str]] = []
        for shard in sorted(os.listdir(shards_dir)):
            if not shard.endswith(".h5"):
                continue
            sp = os.path.join(shards_dir, shard)
            with h5py.File(sp, "r", libver="latest") as hf:
                if "volumes" not in hf:
                    continue
                for key in hf["volumes"].keys():
                    self._samples.append((sp, key))
        if not self._samples:
            raise RuntimeError(f"No samples under {shards_dir}")

        self.rng = random.Random(seed)
        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self):
        return len(self._samples)

    # --------------- I/O + normalization (GLOBAL per volume) ----------------

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
        return vol

    def _robust_norm_qmag_global(self, vol: torch.Tensor) -> torch.Tensor:
        vr, vi = vol[0], vol[1]
        mag = torch.sqrt(vr*vr + vi*vi)
        q = torch.quantile(mag.reshape(-1), self.norm_q)
        if not torch.isfinite(q) or q <= 1e-6:
            return vol
        gain_q = self.norm_target / max(q, 1e-6)
        gain_q = min(gain_q, self.norm_max_gain)
        # prevent clipping: ensure max(|vol|)*gain <= 1.0
        max_abs = vol.abs().amax()
        gain_clip = (1.0 / max_abs).item() if max_abs > 0 else gain_q
        gain = min(gain_q, gain_clip)      # choose the no-clip gain
        return vol * gain                   # no clamp here

    # ---------------- spatial crop helpers (match prior random 80x80 behavior) ----------------
    def _ensure_min_spatial(self, vol: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
        """
        Reflect-pad H/W to at least (min_h, min_w). vol is [2, T, H, W].
        We treat (T*2) as 'channels' to use 4D reflect pad cleanly.
        """
        C, T, H, W = vol.shape
        pad_h = max(0, min_h - H)
        pad_w = max(0, min_w - W)
        if pad_h == 0 and pad_w == 0:
            return vol

        # to [1, T*C, H, W]
        x = vol.permute(1, 0, 2, 3).contiguous().view(1, T * C, H, W)
        ph_top = pad_h // 2
        ph_bot = pad_h - ph_top
        pw_left = pad_w // 2
        pw_right = pad_w - pw_left
        x = F.pad(x, (pw_left, pw_right, ph_top, ph_bot), mode="reflect")
        Hp, Wp = x.shape[-2], x.shape[-1]
        # back to [2, T, Hp, Wp]
        x = x.view(T, C, Hp, Wp).permute(1, 0, 2, 3).contiguous()
        return x

    def _random_crop_hw(self, vol: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
        """Random spatial crop to (ch, cw) on [2, T, H, W]."""
        vol = self._ensure_min_spatial(vol, ch, cw)
        _, _, H, W = vol.shape
        y0 = 0 if H == ch else self.rng.randint(0, H - ch)
        x0 = 0 if W == cw else self.rng.randint(0, W - cw)
        return vol[:, :, y0:y0 + ch, x0:x0 + cw].contiguous()

    # ---------------- temporal helpers (preserve prior logic) ----------------

    @staticmethod
    def _oddify(vol: torch.Tensor) -> torch.Tensor:
        # [2, T, H, W] -> drop last frame if even
        T = vol.shape[1]
        return vol if (T % 2 == 1) else vol[:, :T - 1]

    def _pick_train_clip_indices(self, T: int, L: int) -> List[int]:
        """Random contiguous window of length L; circular wrap if needed."""
        if T >= L:
            start = self.rng.randint(0, T - L)
            return list(range(start, start + L))
        # wrap-around
        start = self.rng.randint(0, T - 1)
        return [(start + i) % T for i in range(L)]

    # ---------------- dataset API ----------------

    def __getitem__(self, idx: int):
        shard_path, key = self._samples[idx]
        vol = self._load_volume(shard_path, key)  # [2, T, H, W] (GLOBAL-normed)

        if self.split == "train":
            # temporal choice
            if self.stage_mode == "pretrain_2d":
                # single random frame
                t = self.rng.randint(0, vol.shape[1] - 1)
                clip = vol[:, t:t + 1]                         # [2,1,H,W]
            else:
                # videos mode: choose L (fixed or from t_choices)
                L = self.fixed_train_L if self.fixed_train_L is not None else self.rng.choice(self.t_choices)
                assert L % 2 == 1
                idxs = self._pick_train_clip_indices(vol.shape[1], L)
                clip = vol[:, idxs]                             # [2,L,H,W]

            # spatial crop to 80x80 (or configured)
            clip = self._random_crop_hw(clip, self.crop_h, self.crop_w)  # [2,L,80,80] or [2,1,80,80]
            return clip
        else:
            # -----------------------------
            # val/test: respect stage_mode
            # -----------------------------
            if self.stage_mode == "pretrain_2d":
                # return a single (middle) frame: [2,1,H,W]
                T = vol.shape[1]
                t0 = T // 2
                return vol[:, t0:t0+1].contiguous()
            else:
                # videos mode: full video with odd T
                return self._oddify(vol).contiguous()

    def close(self):
        for p, f in list(self._h5_cache.items()):
            try:
                f.close()
            except:
                pass
        self._h5_cache.clear()

    def __del__(self):
        try:
            self.close()
        except:
            pass
