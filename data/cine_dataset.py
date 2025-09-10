import os, json, h5py, torch, random
import numpy as np
from typing import List, Tuple, Dict

class CINEDataset(torch.utils.data.Dataset):
    """
    HDF5-backed dataset for cardiac CINE MRI with patching for training
    and full-resolution clips for validation/test.

    Returns:
      - stage_mode="pretrain_2d":
          * train -> [2, 1, 80, 80]
          * val/test -> [2, 1, H, W]          (full-res single frame, no padding)
      - stage_mode="videos":
          * train -> [2, 11, 80, 80]
          * val/test -> [2, L_odd, H, W]      (full-res full time, no padding)

    Notes:
      * Data in H5 is float32 nominally in [-1,1], layout [2,T,H,W].
      * Training spatial patch = 80×80. Temporal patch length = 11 (always).
      * NEW (defaults enabled):
          - Robust per-volume magnitude normalization ("qmag"): scale so mag q≈0.995 maps
            to ~0.9, same scalar applied to both channels. Limits extreme gains.
          - Non-wrap temporal windows at train time when T ≥ patch_t (still wraps if T < patch_t).
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 stage_mode: str = "videos",
                 frame_selection: str = None,
                 seed: int = 123,
                 patch_t: int = 11,
                 patch_h: int = 80,
                 patch_w: int = 80,
                 trunc_std_scale: float = 1.5,
                 flip_p: float = 0.5,
                 # ---- new knobs (safe defaults) ----
                 normalize: str = "qmag",     # "none" or "qmag"
                 norm_q: float = 0.995,       # quantile used for robust scaling
                 norm_target: float = 0.90,   # target magnitude for that quantile
                 norm_max_gain: float = 1.5,  # cap scale-up to avoid crazy boosts
                 temporal_wrap_train: bool = False  # use non-wrap windows when possible
                 ):
        assert split in ("train","val","test")
        assert stage_mode in ("pretrain_2d","videos")
        self.root = root
        self.split = split
        self.stage_mode = stage_mode

        self.patch_t = int(patch_t)   # 11
        self.patch_h = int(patch_h)   # 80
        self.patch_w = int(patch_w)   # 80
        self.trunc_std_scale = float(trunc_std_scale)
        self.flip_p = float(flip_p)

        # new options
        self.normalize = str(normalize).lower()
        assert self.normalize in ("none", "qmag")
        self.norm_q = float(norm_q)
        self.norm_target = float(norm_target)
        self.norm_max_gain = float(norm_max_gain)
        self.temporal_wrap_train = bool(temporal_wrap_train)

        self.split_dir = os.path.join(root, split)
        self.shards_dir = os.path.join(self.split_dir, "shards")

        meta_path = os.path.join(self.split_dir, "split_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"split_meta.json not found at {meta_path}. Run preprocessing first.")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self._samples: List[Tuple[str, str]] = []
        for shard in sorted(os.listdir(self.shards_dir)):
            if not shard.endswith(".h5"): continue
            shard_path = os.path.join(self.shards_dir, shard)
            with h5py.File(shard_path, "r", libver="latest") as hf:
                if "volumes" not in hf: continue
                for key in hf["volumes"].keys():
                    self._samples.append((shard_path, key))
        if not self._samples:
            raise RuntimeError(f"No samples found under {self.shards_dir}")

        if frame_selection is None:
            self.frame_selection = "random" if split == "train" else "middle"
        else:
            assert frame_selection in ("random","middle")
            self.frame_selection = frame_selection

        self.rng = random.Random(seed)
        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self._samples)

    # ---------- HDF5 ----------
    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest", swmr=True)
            self._h5_cache[path] = f
        return f

    def _load_volume(self, shard_path: str, group_name: str) -> torch.Tensor:
        hf = self._get_file(shard_path)
        ds = hf["volumes"][group_name]
        arr = ds[()].astype("float32", copy=False)  # [2,T,H,W] nominally in [-1,1]
        np.clip(arr, -1.0, 1.0, out=arr)
        vol = torch.from_numpy(arr)

        # ---- robust per-volume mag normalization (same scalar on real/imag) ----
        if self.normalize == "qmag":
            vol = self._apply_robust_mag_norm(vol)

        return vol

    # ---------- helpers ----------
    @staticmethod
    def _ensure_min_hw(x: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
        _, T, H, W = x.shape
        H2, W2 = max(H, min_h), max(W, min_w)
        if H2 == H and W2 == W:
            return x
        out = x.new_zeros((2, T, H2, W2))
        out[..., :H, :W] = x
        return out

    @staticmethod
    def _oddify_time_full(x: torch.Tensor) -> torch.Tensor:
        _, T, _, _ = x.shape
        if T % 2 == 1 or T <= 1:
            return x
        return x[:, :T-1]

    def _time_indices_circular(self, T: int, L: int):
        start = self.rng.randint(0, max(T - 1, 0))
        return [ (start + i) % T for i in range(L) ]

    def _time_indices_train(self, T: int, L: int):
        """Prefer a contiguous, non-wrapping window at train time when possible."""
        if self.temporal_wrap_train or T < L:
            return self._time_indices_circular(T, L)
        # non-wrapping continuous window
        start = self.rng.randint(0, T - L)
        return list(range(start, start + L))

    def _pick_frame_index(self, T: int) -> int:
        if self.frame_selection == "middle" or self.split != "train":
            return T // 2
        return self.rng.randint(0, max(0, T - 1))

    def _truncnorm_centered(self, H: int, W: int) -> Tuple[int, int]:
        # top
        top_mu  = (H - self.patch_h) / 2.0
        top_sig = max((H - self.patch_h) / 6.0, 1e-6) * (self.trunc_std_scale / 2.0)
        top_lo, top_hi = 0.0, max(H - self.patch_h, 0)
        for _ in range(8):
            t = self.rng.gauss(top_mu, top_sig)
            if top_lo <= t <= top_hi:
                top = int(t); break
        else:
            top = int(min(max(top_mu, top_lo), top_hi))
        # left (pre-roll space)
        left_mu  = (W - self.patch_w) / 2.0
        left_sig = max((W - self.patch_w) / 6.0, 1e-6) * (self.trunc_std_scale)
        left_lo, left_hi = -(self.patch_w - 1), (W - 1)
        for _ in range(8):
            l = self.rng.gauss(left_mu, left_sig)
            if left_lo <= l <= left_hi:
                left = int(l); break
        else:
            left = int(min(max(left_mu, left_lo), left_hi))
        return top, left

    # ---- robust normalization (same scalar on both channels) ----
    def _apply_robust_mag_norm(self, vol: torch.Tensor) -> torch.Tensor:
        # vol: [2,T,H,W] (CPU)
        vr, vi = vol[0], vol[1]
        mag = torch.sqrt(vr * vr + vi * vi)          # [T,H,W]
        flat = mag.reshape(-1)
        if flat.numel() == 0:
            return vol
        q = torch.quantile(flat, self.norm_q)
        q = float(q)
        if not np.isfinite(q) or q <= 1e-6:
            return vol  # degenerate; skip
        gain = self.norm_target / max(q, 1e-6)
        # cap extreme scale-up; allow scale-down freely
        if gain > self.norm_max_gain:
            gain = self.norm_max_gain
        vol = (vol * gain).clamp_(-1.0, 1.0)
        return vol

    # ---------- main ----------
    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_path, group_name = self._samples[idx]
        vol = self._load_volume(shard_path, group_name)  # [2,T,H,W]
        _, T, H, W = vol.shape

        if self.split == "train":
            # TRAIN (patchified)
            if self.stage_mode == "pretrain_2d":
                vol = self._ensure_min_hw(vol, self.patch_h, self.patch_w)
                t_idx = self._pick_frame_index(vol.shape[1])
                clip = vol[:, t_idx:t_idx+1]  # [2,1,H,W]
                top, left = self._truncnorm_centered(clip.shape[-2], clip.shape[-1])
                y0 = max(0, min(top,  clip.shape[-2] - self.patch_h))
                x0 = max(0, min(left, clip.shape[-1] - self.patch_w))
                clip = clip[..., y0:y0 + self.patch_h, x0:x0 + self.patch_w]
            else:
                vol = self._ensure_min_hw(vol, self.patch_h, self.patch_w)
                idxs = self._time_indices_train(vol.shape[1], self.patch_t)  # prefer non-wrap
                clip = vol[:, idxs]  # [2,11,H,W]
                top, left = self._truncnorm_centered(clip.shape[-2], clip.shape[-1])
                if left != 0:
                    clip = torch.roll(clip, shifts=-left, dims=-1)
                y0 = max(0, min(top, clip.shape[-2] - self.patch_h))
                clip = clip[..., y0:y0 + self.patch_h, 0:self.patch_w]      # [2,11,80,80]

            if self.stage_mode == "pretrain_2d":
                # single frame: compute mag on that frame
                mag = torch.sqrt(clip[0]**2 + clip[1]**2).squeeze(0)  # [H,W]
            else:
                # videos: use the middle frame for cheap gating
                tmid = clip.shape[1] // 2
                mag = torch.sqrt(clip[0, tmid]**2 + clip[1, tmid]**2) # [H,W]

            ok = (mag.mean() >= 0.03) and (mag.std(unbiased=False) >= 0.03)  # thresholds ~ a hair above your q5%
            tries = 0
            while (not ok) and (tries < 4):  # bounded retries
                tries += 1
                # resample a new (top,left) and crop (same logic as above)
                top, left = self._truncnorm_centered(vol.shape[-2], vol.shape[-1])
                y0 = max(0, min(top,  vol.shape[-2] - self.patch_h))
                x0 = max(0, min(left, vol.shape[-1] - self.patch_w))
                if self.stage_mode == "pretrain_2d":
                    t_idx = self._pick_frame_index(vol.shape[1])
                    clip = vol[:, t_idx:t_idx+1, y0:y0+self.patch_h, x0:x0+self.patch_w]
                    mag  = torch.sqrt(clip[0]**2 + clip[1]**2).squeeze(0)
                else:
                    idxs = self._time_indices_train(vol.shape[1], self.patch_t)
                    clip = vol[:, idxs, y0:y0+self.patch_h, x0:x0+self.patch_w]
                    tmid = clip.shape[1] // 2
                    mag  = torch.sqrt(clip[0, tmid]**2 + clip[1, tmid]**2)

                ok = (mag.mean() >= 0.03) and (mag.std(unbiased=False) >= 0.03)

            # flips
            if self.rng.random() < self.flip_p:
                clip = torch.flip(clip, dims=[-1])
            # if self.rng.random() < self.flip_p:
                # clip = torch.flip(clip, dims=[-2])
            return clip.contiguous()

        # VAL/TEST (full-res, no padding)
        if self.stage_mode == "pretrain_2d":
            t_idx = self._pick_frame_index(T)   # middle by default for val/test
            clip = vol[:, t_idx:t_idx+1]        # [2,1,H,W]
            return clip.contiguous()
        else:
            x = self._oddify_time_full(vol)     # [2,T_odd,H,W], no padding
            return x.contiguous()

    # ---------- cleanup ----------
    def close(self):
        for p, f in list(self._h5_cache.items()):
            try: f.close()
            except: pass
        self._h5_cache.clear()

    def __del__(self):
        try: self.close()
        except: pass
