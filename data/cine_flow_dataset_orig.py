# data/cine_latent_or_raw_dataset.py
import os, json, h5py, torch
from typing import List, Tuple, Dict, Any, Optional

# We import CINEDataset for val/test pass-through.
from data.cine_dataset import CINEDataset

class CINETrainLatentsValRawDataset(torch.utils.data.Dataset):
    """
    Split-dependent behavior:
      • train:   returns precomputed latents z [Cz, nt, H', W'] produced by the precompute script.
      • val/test:delegates to CINEDataset(split, stage_mode='videos'), returning the FULL normalized video
                  (odd T) in pixel space: x [2, T, H, W].

    Args:
      latents_root: directory that contains <split>/latents_meta.json and latent_shards/*.h5
      raw_root:     original raw CINE root (with <split>/shards/*.h5) for val/test pass-through
      split:        'train' | 'val' | 'test'
      fixed_L:      if set, only keep train latent items with that L
      include_sources: optional whitelist of (src_shard, src_key) for train
      cineds_kwargs: kwargs forwarded to CINEDataset for val/test (e.g. normalize='qmag')
    """
    def __init__(
        self,
        latents_root: str,
        raw_root: str,
        split: str = "train",
        fixed_L: Optional[int] = None,
        include_sources: Optional[List[Tuple[str,str]]] = None,
        cineds_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.latents_root = latents_root
        self.raw_root = raw_root
        self.cineds_kwargs = dict(cineds_kwargs or {})

        if split == "train":
            meta_path = os.path.join(latents_root, split, "latents_meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(f"latents meta not found: {meta_path}")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            base = os.path.join(latents_root, split, "latent_shards")

            items_all = meta["items"]
            items: List[Dict[str,Any]] = []
            for it in items_all:
                if fixed_L is not None and int(it.get("L", -1)) != int(fixed_L):
                    continue
                if include_sources is not None:
                    k = (it.get("src_shard",""), it.get("src_key",""))
                    if k not in include_sources:
                        continue
                it2 = dict(it)
                it2["_h5_path"] = os.path.join(base, it["ds"])
                items.append(it2)

            if not items:
                raise RuntimeError("No latent items matched the filters.")
            self.items = items
            self._open_files: Dict[str, h5py.File] = {}
            self._valtest = None
        else:
            # val/test: direct CINEDataset pass-through; stage_mode='videos' returns the full normalized volume
            self._open_files = {}
            self.items = []
            self._valtest = CINEDataset(
                root=raw_root,
                split=split,
                stage_mode="videos",
                **self.cineds_kwargs
            )

    def __len__(self):
        if self.split == "train":
            return len(self.items)
        return len(self._valtest)

    def _get_h5(self, path: str) -> h5py.File:
        f = self._open_files.get(path)
        if f is None:
            f = h5py.File(path, "r", libver="latest")
            self._open_files[path] = f
        return f

    def __getitem__(self, idx: int):
        if self.split == "train":
            it = self.items[idx]
            f = self._get_h5(it["_h5_path"])
            g = f[it["name"]]
            z = torch.from_numpy(g["z"][()])  # [Cz, nt, H', W']
            return z, it
        else:
            # (x_2THW) full normalized video with odd T, exactly as CINEDataset(videos) does.
            x = self._valtest[idx]
            return x

    def close(self):
        for p, f in list(self._open_files.items()):
            try: f.close()
            except: pass
        self._open_files.clear()
        if self._valtest is not None and hasattr(self._valtest, "close"):
            self._valtest.close()

    def __del__(self):
        try: self.close()
        except: pass
