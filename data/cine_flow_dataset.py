import os, json, random, h5py
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

# For val/test we keep your existing dataset (unchanged behavior)
from data.cine_dataset import CINEDataset  # returns [2, L, H, W] with odd L

class CINEFlowMatchLatentDataset(Dataset):
    """
    TRAIN: serves precomputed 7-frame latent windows:
      returns a *list* of tensors, each [Cz, 7, P, H_lat, W_lat]
      (list length = num_time_samples)
    VAL/TEST: falls back to CINEDataset, returning what it normally returns
      (so validation remains full VAE pipeline).
    """
    def __init__(self, root: str, split: str, num_time_samples: int = 1, patch_batch: int = 8):
        self.root = root
        self.split = split
        self.num_time_samples = int(num_time_samples)
        self.patch_batch = int(patch_batch)

        if split == "train":
            split_dir = os.path.join(root, split)
            meta_path = os.path.join(split_dir, "latents_meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Missing {meta_path}. Run tools/precompute_latents.py first.")
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            # index by source (so each __getitem__ can sample multiple windows from same video)
            groups: Dict[Tuple[str,str], List[Tuple[str,str]]] = {}
            for it in self.meta["items"]:
                src = (it["source_shard"], it["source_key"])
                groups.setdefault(src, []).append((it["ds"], it["name"]))
            self.sources = list(groups.keys())
            self.by_source = groups
            self.shards_cache: Dict[str, h5py.File] = {}
        else:
            # unchanged behavior for val/test
            self._cine = CINEDataset(root=root, split=split, normalize="qmag", stage_mode="videos")

    def __len__(self):
        return len(self.sources) if self.split == "train" else len(self._cine)

    def _open(self, shard_file: str) -> h5py.File:
        path = os.path.join(self.root, self.split, "latent_shards", shard_file)
        f = self.shards_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest")
            self.shards_cache[path] = f
        return f

    def __getitem__(self, idx: int):
        if self.split != "train":
            return self._cine[idx]  # [2, L, H, W] (unchanged)
        src = self.sources[idx]
        pool = self.by_source[src]
        k = min(self.num_time_samples, len(pool))
        picks = random.sample(pool, k=k)
        clips = []
        for ds, name in picks:
            f = self._open(ds)
            z = torch.from_numpy(f[name]["z"][()])
            num_patches = z.shape[2]
            if self.patch_size > 0:
                patch_dim_mid = num_patches // 2
                z = z[:, :, patch_dim_mid - self.patch_batch // 2:patch_dim_mid + self.patch_batch // 2, :, :]
                
            clips.append(z)
        return clips  # ragged collate keeps lists per item
