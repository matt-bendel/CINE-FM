# data/cine_dataset.py
import os
import json
import h5py
import torch
import random
import numpy as np
from typing import List, Tuple, Dict, Optional


class CINEDataset(torch.utils.data.Dataset):
    """
    VAE dataset (stage-aware).

    Disk layout per split:
      root/{train,val,test}/
        split_meta.json
        shards/*.h5     # each with group "volumes" -> datasets [2, T, H, W] in [-1,1]

    Returns (depends on stage_mode):
      - pretrain_2d:
          train:     Tensor [2, 1, H, W]  (random frame)
          val/test:  Tensor [2, 1, H, W]  (center frame)
      - videos:
          train:     Tensor [2, L, H, W]  (fixed odd L = train_clip_len, contiguous window; wrap if T < L)
          val/test:  Tensor [2, T, H, W]  (FULL video, unchanged length)

    Normalization:
      Global, per-volume complex-magnitude quantile scaling is applied BEFORE any slicing
      if normalize == "qmag".
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        stage_mode: str = "pretrain_2d",   # "pretrain_2d" or "videos"
        seed: int = 123,
        # --- global normalization (per-volume via magnitude quantile) ---
        normalize: str = "qmag",           # "qmag" or "none"
        norm_q: float = 0.995,
        norm_target: float = 0.90,
        norm_max_gain: float = 1.5,
        # --- training clip length for stage_mode == "videos" ---
        train_clip_len: int = 11,          # must be odd; will be forced odd if needed
    ):
        assert split in ("train", "val", "test")
        assert stage_mode in ("pretrain_2d", "videos")
        self.root = root
        self.split = split
        self.stage_mode = stage_mode

        # normalization settings
        self.normalize = str(normalize).lower()
        assert self.normalize in ("none", "qmag")
        self.norm_q = float(norm_q)
        self.norm_target = float(norm_target)
        self.norm_max_gain = float(norm_max_gain)

        # fixed train clip length for videos mode
        L = int(train_clip_len)
        if L < 1:
            L = 1
        if L % 2 == 0:
            L += 1
        self.train_clip_len = L

        # dataset index
        self.split_dir = os.path.join(root, split)
        self.shards_dir = os.path.join(self.split_dir, "shards")
        meta_path = os.path.join(self.split_dir, "split_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"split_meta.json not found at {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self._samples: List[Tuple[str, str]] = []
        if not os.path.isdir(self.shards_dir):
            raise FileNotFoundError(f"shards dir not found: {self.shards_dir}")
        for shard in sorted(os.listdir(self.shards_dir)):
            if not shard.endswith(".h5"):
                continue
            sp = os.path.join(self.shards_dir, shard)
            with h5py.File(sp, "r", libver="latest") as hf:
                if "volumes" not in hf:
                    continue
                for key in hf["volumes"].keys():
                    self._samples.append((sp, key))
        if not self._samples:
            raise RuntimeError(f"No samples under {self.shards_dir}")

        # rng
        self.rng = random.Random(seed)

        # h5 handle cache
        self._h5_cache: Dict[str, h5py.File] = {}

    # ------------- h5 helpers -------------

    def __len__(self) -> int:
        return len(self._samples)

    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None:
            # SWMR for parallel readers
            f = h5py.File(path, "r", libver="latest", swmr=True)
            self._h5_cache[path] = f
        return f

    def close(self):
        for p, f in list(self._h5_cache.items()):
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------- normalization -------------

    def _robust_norm(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Global, per-volume normalization using complex magnitude quantile.

        vol: [2, T, H, W] with values in [-1, 1].
             Compute mag = sqrt(r^2 + i^2) across ENTIRE volume (all frames),
             take q = quantile(norm_q), and scale real/imag channels by
             gain = norm_target / q, limited to norm_max_gain.
        """
        if not (vol.ndim == 4 and vol.shape[0] == 2):
            return vol

        vr, vi = vol[0], vol[1]
        mag = torch.sqrt(vr * vr + vi * vi)
        flat = mag.reshape(-1)
        if flat.numel() == 0:
            return vol

        q = torch.quantile(flat, self.norm_q)
        q = float(q)
        if not np.isfinite(q) or q <= 1e-6:
            return vol

        gain = self.norm_target / max(q, 1e-6)
        gain = min(gain, self.norm_max_gain)

        return (vol * gain).clamp_(-1, 1)

    def _load_volume(self, shard_path: str, group_name: str) -> torch.Tensor:
        """
        Load full volume [2, T, H, W] then apply *global per-volume* normalization
        (if enabled) BEFORE any slicing/windowing.
        """
        hf = self._get_file(shard_path)
        vol = torch.from_numpy(hf["volumes"][group_name][()]).float()  # [2, T, H, W]
        vol.clamp_(-1, 1)
        if self.normalize == "qmag":
            vol = self._robust_norm(vol)
        return vol

    # ------------- item builders -------------

    def _get_pretrain_2d(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Return a single frame [2, 1, H, W].
        Train: random frame
        Val/Test: center frame
        """
        _, T, _, _ = vol.shape
        if self.split == "train":
            t = self.rng.randint(0, T - 1) if T > 1 else 0
        else:
            t = T // 2
        return vol[:, t : t + 1].contiguous()

    def _get_videos_train_fixed(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Fixed-L training clip: [2, L, H, W] with L = self.train_clip_len (odd).
        If T >= L: random contiguous window.
        If T < L: wrap-around indexing to reach L frames.
        """
        _, T, _, _ = vol.shape
        L = self.train_clip_len

        if T >= L:
            start = self.rng.randint(0, max(0, T - L))
            clip = vol[:, start : start + L]
        else:
            if T == 1:
                clip = vol.repeat(1, L, 1, 1)
            else:
                start = self.rng.randint(0, T - 1)
                idxs = [(start + i) % T for i in range(L)]
                clip = vol[:, idxs]
        return clip.contiguous()

    def _get_videos_eval_full(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: return FULL video volume [2, T, H, W], unchanged.
        """
        return vol.contiguous()

    # ------------- main -------------

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_path, key = self._samples[idx]
        vol = self._load_volume(shard_path, key)  # [2, T, H, W]

        if self.stage_mode == "pretrain_2d":
            return self._get_pretrain_2d(vol)

        # videos
        if self.split == "train":
            return self._get_videos_train_fixed(vol)
        else:
            # val/test: full volume (no frame dropped)
            return self._get_videos_eval_full(vol)
