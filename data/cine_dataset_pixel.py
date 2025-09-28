# data/cine_dataset_pixel.py
import os, glob, h5py, random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple, Dict, Optional

class CINEPixelDataset(torch.utils.data.Dataset):
    """
    Pixel-space CINE MRI dataset that mirrors your colleague's PKL loader,
    but sources samples from HDF5 shards.

    Train:
      • Returns [2, 8, 64, 64]
        - time: random circular 8-frame window
        - space: truncnorm-centered crop with horizontal "roll" (circular) on W, then crop
        - flips: random horizontal and vertical
      • NO normalization (values are used as-is).

    Val/Test:
      • Returns full volume [2, T, H, W] (no crop, no norm).
    """

    # --------------------- init ---------------------
    def __init__(
        self,
        root: str,
        split: str = "train",     # "train" | "val" | "test"
        # training geometry (match colleague)
        t_frame: int = 8,
        t_H: int = 64,
        t_W: int = 64,
        # truncnorm concentration (match colleague defaults)
        std_scale_top: float = 1.5,   # used as *0.5* on top_std like colleague's code
        std_scale_left: float = 1.5,
        seed: int = 123,
        **kwargs,                    # ignore any extra keys
    ):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split

        # train geometry
        self.t_frame = int(t_frame)
        self.t_H = int(t_H)
        self.t_W = int(t_W)

        # truncnorm scales
        self.std_scale_top = float(std_scale_top)
        self.std_scale_left = float(std_scale_left)

        # discover shards
        split_dir = os.path.join(root, split)
        shards_dir = os.path.join(split_dir, "shards")
        if not os.path.isdir(shards_dir):
            raise FileNotFoundError(f"Expected shards at {shards_dir}")

        self._samples: List[Tuple[str, str]] = []
        for sp in sorted(glob.glob(os.path.join(shards_dir, "*.h5"))):
            try:
                with h5py.File(sp, "r", libver="latest") as hf:
                    if "volumes" not in hf:
                        continue
                    for key in hf["volumes"].keys():
                        # each item expected shape [2, T, H, W]
                        self._samples.append((sp, key))
            except Exception as e:
                print(f"[CINEPixelDataset] Skipping shard {sp}: {e}")

        if not self._samples:
            raise RuntimeError(f"No 'volumes/*' datasets found under {shards_dir}")

        self.rng = random.Random(seed)
        self._h5_cache: Dict[str, h5py.File] = {}

    def __len__(self):
        return len(self._samples)

    # --------------------- HDF5 helpers ---------------------
    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest", swmr=True)
            self._h5_cache[path] = f
        return f

    def _load_volume(self, shard_path: str, key: str) -> torch.Tensor:
        hf = self._get_file(shard_path)
        arr = hf["volumes"][key][()]  # numpy array, [2, T, H, W]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        x = torch.from_numpy(arr).float()
        # DO NOT normalize; DO NOT clamp (match colleague)
        return x

    # --------------------- truncnorm sampler (no SciPy) ---------------------
    @staticmethod
    def _truncnorm_int(mean: float, std: float, a: float, b: float, rng: random.Random) -> int:
        """Simple rejection sampler for truncated normal, returns int."""
        if std <= 1e-8:
            return int(round(max(a, min(b, mean))))
        # a few tries is fine for our bounds
        for _ in range(100):
            z = rng.gauss(0.0, 1.0)
            v = mean + std * z
            if a <= v <= b:
                return int(v)
        # fallback to clamped mean if rejection failed
        return int(round(max(a, min(b, mean))))

    def _sample_top_left(self, H: int, W: int) -> Tuple[int, int]:
        # --- top ---
        # colleague: top_mean = (H - t_H)/2 ; top_std = (H - t_H)/6 ; *= (std_scale/2)
        top_mean = (H - self.t_H) / 2.0
        top_std = (H - self.t_H) / 6.0 * (self.std_scale_top / 2.0)
        top_a, top_b = 0.0, float(max(0, H - self.t_H))
        top = self._truncnorm_int(top_mean, max(1e-6, top_std), top_a, top_b, self.rng)

        # --- left ---
        # colleague: left_mean = (W - t_W)/2 ; left_std = (W - t_W)/6 ; *= std_scale
        # bounds: a = -(t_W - 1), b = W - 1  (circular roll then crop at x=0)
        left_mean = (W - self.t_W) / 2.0
        left_std = (W - self.t_W) / 6.0 * self.std_scale_left
        left_a, left_b = float(-(self.t_W - 1)), float(W - 1)
        left = self._truncnorm_int(left_mean, max(1e-6, left_std), left_a, left_b, self.rng)

        return int(top), int(left)

    # --------------------- dataset API ---------------------
    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_path, key = self._samples[idx]
        vol = self._load_volume(shard_path, key)   # [2, T, H, W]

        if self.split == "train":
            # ----- time: random circular crop to length 8 -----
            _, T, H, W = vol.shape
            t0 = self.rng.randint(0, max(0, T - self.t_frame))
            idxs = [ (t0 + i) % T for i in range(self.t_frame) ]
            clip = vol[:, idxs, :, :]  # [2, 8, H, W]

            # ----- space: roll + truncnorm-centered crop to 64x64 -----
            top, left = self._sample_top_left(H, W)
            # circular roll along W by -left (match colleague)
            clip = torch.roll(clip, shifts=-left, dims=-1)
            # then crop at (top, 0)
            clip = TF.crop(clip, top, 0, self.t_H, self.t_W)  # [2, 8, 64, 64]

            # random flips
            if self.rng.random() > 0.5:
                clip = torch.flip(clip, dims=[-1])  # horizontal
            if self.rng.random() > 0.5:
                clip = torch.flip(clip, dims=[-2])  # vertical

            # final assert (useful if source sizes vary)
            assert clip.shape == (2, self.t_frame, self.t_H, self.t_W), \
                f"Got {clip.shape}, expected (2,{self.t_frame},{self.t_H},{self.t_W})"
            return clip

        # val/test: return full volume (no crop, no norm)
        return vol.contiguous()

    # --------------------- cleanup ---------------------
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
