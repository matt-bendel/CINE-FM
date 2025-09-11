# data/cine_fm_dataset.py
import os, json, h5py, torch, random
import numpy as np
from typing import List, Tuple, Dict

class CINEFlowMatchDataset(torch.utils.data.Dataset):
    """
    For latent flow matching (unconditional).
    Train:
      __getitem__ -> List[ Tensor [2, T, H, W] ] for multiple random T (odd in [1,11]).
    Val/Test:
      __getitem__ -> Tensor [2, L_odd, H, W] full video (no patching here).
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        seed: int = 123,
        normalize: str = "qmag",
        norm_q: float = 0.995,
        norm_target: float = 0.90,
        norm_max_gain: float = 1.5,
        t_choices: List[int] = None,    # default odd 1..11
        num_time_samples: int = 1,      # how many different T per item (train only)
    ):
        assert split in ("train","val","test")
        self.root = root
        self.split = split

        self.normalize = str(normalize).lower()
        assert self.normalize in ("none", "qmag")
        self.norm_q = float(norm_q)
        self.norm_target = float(norm_target)
        self.norm_max_gain = float(norm_max_gain)

        self.split_dir = os.path.join(root, split)
        self.shards_dir = os.path.join(self.split_dir, "shards")
        meta_path = os.path.join(self.split_dir, "split_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"split_meta.json not found at {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self._samples: List[Tuple[str, str]] = []
        for shard in sorted(os.listdir(self.shards_dir)):
            if not shard.endswith(".h5"): continue
            sp = os.path.join(self.shards_dir, shard)
            with h5py.File(sp, "r", libver="latest") as hf:
                if "volumes" not in hf: continue
                for key in hf["volumes"].keys():
                    self._samples.append((sp, key))
        if not self._samples:
            raise RuntimeError(f"No samples under {self.shards_dir}")

        self.rng = random.Random(seed)
        self._h5_cache: Dict[str, h5py.File] = {}

        if t_choices is None:
            self.t_choices = [1,3,5,7,9,11]
        else:
            self.t_choices = [int(t) for t in t_choices if (t % 2 == 1 and 1 <= t <= 11)]
            assert len(self.t_choices) > 0, "t_choices must include odd in [1,11]"
        self.num_time_samples = int(num_time_samples)

    def __len__(self):
        return len(self._samples)

    def _get_file(self, path: str) -> h5py.File:
        f = self._h5_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest", swmr=True)
            self._h5_cache[path] = f
        return f

    def _load_volume(self, shard_path, group_name) -> torch.Tensor:
        hf = self._get_file(shard_path)
        vol = torch.from_numpy(hf["volumes"][group_name][()]).float()  # [2,T,H,W] in [-1,1]
        vol.clamp_(-1, 1)
        return self._robust_norm(vol) if self.normalize == "qmag" else vol

    def _robust_norm(self, vol: torch.Tensor) -> torch.Tensor:
        vr, vi = vol[0], vol[1]
        mag = torch.sqrt(vr*vr + vi*vi)
        flat = mag.reshape(-1)
        if flat.numel() == 0:
            return vol
        q = torch.quantile(flat, self.norm_q)
        q = float(q)
        if not np.isfinite(q) or q <= 1e-6: return vol
        gain = self.norm_target / max(q, 1e-6)
        gain = min(gain, self.norm_max_gain)
        return (vol * gain).clamp_(-1, 1)

    @staticmethod
    def _oddify(x: torch.Tensor) -> torch.Tensor:
        # [2,T,H,W] -> drop last frame if even
        T = x.shape[1]
        return x if (T % 2 == 1) else x[:, :T-1]

    def __getitem__(self, idx: int):
        shard_path, key = self._samples[idx]
        vol = self._load_volume(shard_path, key)  # [2,T,H,W]
        _, T, H, W = vol.shape

        if self.split == "train":
            out: List[torch.Tensor] = []
            # draw num_time_samples choices (with replacement)
            for _ in range(self.num_time_samples):
                L = self.rng.choice(self.t_choices)
                if L == 1:
                    # pick middle-ish single frame
                    t0 = T // 2
                    out.append(vol[:, t0:t0+1].contiguous())
                else:
                    # contiguous window of length L (wrap if too short)
                    if T >= L:
                        start = self.rng.randint(0, T - L)
                        clip = vol[:, start:start+L]
                    else:
                        # circular gather
                        idxs = [(i % T) for i in range(self.rng.randint(0, T-1), self.rng.randint(0, T-1)+L)]
                        clip = vol[:, idxs]
                    out.append(clip.contiguous())
            return out  # list[Tensor[2,L,H,W]]

        # val/test: full video, ensure odd T
        return self._oddify(vol).contiguous()

    def close(self):
        for p, f in list(self._h5_cache.items()):
            try: f.close()
            except: pass
        self._h5_cache.clear()

    def __del__(self):
        try: self.close()
        except: pass
