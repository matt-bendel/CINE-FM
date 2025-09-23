import os, json, h5py, random
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from data.cine_dataset import CINEDataset

class CINEFlowMatchLatentDataset(Dataset):
    """
    TRAIN:
      - Loads latents saved by tools/precompute_train_latents_like_cine.py
      - Each item is a single tensor [Cz, nt, H', W'] (no patch dimension).
    VAL/TEST:
      - Falls back to CINEDataset (unchanged).
    """
    def __init__(self, root: str, split: str):
        self.root = root
        self.split = split

        if split == "train":
            split_dir = os.path.join(root, split)
            meta_path = os.path.join(split_dir, "latents_meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"Missing {meta_path}. Run the precompute script first.")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Flat index over all (shard, name)
            self._index: List[Tuple[str, str]] = [(it["ds"], it["name"]) for it in meta["items"]]
            if not self._index:
                raise RuntimeError("No latent items found.")
            self._h5_cache: Dict[str, h5py.File] = {}
        else:
            self._cine = CINEDataset(root=root, split=split, normalize="qmag", stage_mode="videos")

    def __len__(self):
        return len(self._index) if self.split == "train" else len(self._cine)

    def _open(self, shard_file: str) -> h5py.File:
        path = os.path.join(self.root, self.split, "latent_shards", shard_file)
        f = self._h5_cache.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest")
            self._h5_cache[path] = f
        return f

    def __getitem__(self, idx: int):
        if self.split != "train":
            return self._cine[idx]
        ds, name = self._index[idx]
        f = self._open(ds)
        z = torch.from_numpy(f[name]["z"][()]).float()  # [Cz, nt, H', W']
        return z  # one tensor per item

    def close(self):
        if self.split == "train":
            for p, f in list(self._h5_cache.items()):
                try: f.close()
                except: pass
            self._h5_cache.clear()

    def __del__(self):
        try: self.close()
        except: pass
