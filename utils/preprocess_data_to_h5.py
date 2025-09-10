#!/usr/bin/env python3
import argparse, os, pickle, math, json, mmap, re
from typing import List, Tuple, Dict, Any
import numpy as np
import h5py

SHARD_RE = re.compile(r"^shard_(\d{5})\.h5$")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _choose_chunks(shape: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    # shape = (2, T, H, W)
    c0 = 2
    ct = max(1, min(shape[1], 8))
    ch = max(1, min(shape[2], 128))
    cw = max(1, min(shape[3], 128))
    return (c0, ct, ch, cw)

def _sanitize(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

def _robust_mag_scale(arr: np.ndarray, pctl: float) -> float:
    # arr: [2,T,H,W]; magnitude percentile as scale
    real, imag = arr[0], arr[1]
    mag = np.sqrt(np.maximum(real*real + imag*imag, 0.0))
    mag = np.nan_to_num(mag, nan=0.0, posinf=np.finfo(mag.dtype).max, neginf=0.0)
    scale = np.percentile(mag, pctl)
    return float(max(scale, 1e-6))

def _normalize_to_unit_interval(arr: np.ndarray, pctl: float) -> Tuple[np.ndarray, float]:
    arr = _sanitize(arr.astype(np.float32, copy=False))
    scale = _robust_mag_scale(arr, pctl)
    arr /= scale
    np.clip(arr, -1.0, 1.0, out=arr)
    return arr, scale

def _scan_existing(split_dir: str) -> Dict[str, Any]:
    """
    Returns:
      {
        'has_meta': bool,
        'meta': dict|None,
        'existing_shards': [(path, idx_int, count)],
        'next_shard_num': int,
        'existing_total': int,
        'global_next_index': int
      }
    """
    shards_dir = os.path.join(split_dir, "shards")
    _ensure_dir(shards_dir)

    # Read existing shard files
    existing_shards = []
    for fn in sorted(os.listdir(shards_dir)):
        m = SHARD_RE.match(fn)
        if not m: continue
        idx_int = int(m.group(1))
        path = os.path.join(shards_dir, fn)
        try:
            with h5py.File(path, "r", libver="latest") as f:
                cnt = int(f.attrs.get("count", 0))
        except Exception:
            cnt = 0
        existing_shards.append((path, idx_int, cnt))

    next_shard_num = (max([s[1] for s in existing_shards]) + 1) if existing_shards else 0
    existing_total = sum(s[2] for s in existing_shards)

    meta_path = os.path.join(split_dir, "split_meta.json")
    has_meta = os.path.isfile(meta_path)
    meta = None
    global_next_index = 0

    if has_meta:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Prefer meta-derived next index if consistent, else fall back to count
        shards_meta = meta.get("shards", [])
        if len(shards_meta) > 0:
            last = max(shards_meta, key=lambda s: s["start_idx"])
            global_next_index = int(last["start_idx"]) + int(last["count"])
        else:
            global_next_index = existing_total
    else:
        # No meta; assume we start from 0 and existing_total items are already present
        global_next_index = existing_total

    return {
        "has_meta": has_meta,
        "meta": meta,
        "existing_shards": existing_shards,
        "next_shard_num": next_shard_num,
        "existing_total": existing_total,
        "global_next_index": global_next_index,
        "meta_path": meta_path,
        "shards_dir": shards_dir
    }

def _write_shard(out_path: str, items: List[np.ndarray], start_idx: int,
                 compression: str, pctl: float, norm_method: str):
    assert compression in ("lzf", "gzip", "none")
    comp = None if compression == "none" else compression
    with h5py.File(out_path, "w", libver="latest") as f:
        root = f.create_group("volumes")
        for i, raw in enumerate(items):
            if not (isinstance(raw, np.ndarray) and raw.ndim == 4 and raw.shape[0] == 2):
                raise ValueError(f"Item {i} has invalid shape {getattr(raw,'shape',None)}; expected [2,T,H,W].")
            norm, scale = _normalize_to_unit_interval(raw, pctl)
            ds_name = f"{start_idx + i:08d}"
            chunks = _choose_chunks(norm.shape)
            dset = root.create_dataset(
                ds_name, data=norm, dtype="float32",
                compression=comp, compression_opts=(4 if comp == "gzip" else None),
                chunks=chunks
            )
            dset.attrs["shape"] = norm.shape
            dset.attrs["norm_method"] = norm_method
            dset.attrs["norm_pctl"] = float(pctl)
            dset.attrs["scale_mag"] = float(scale)
        f.attrs["count"] = len(items)

def preprocess(pkl_path: str, out_root: str, split: str, shard_size: int,
               compression: str, start_index: int, pctl: float):
    if split not in ("train", "val", "test"):
        raise ValueError("--split must be train/val/test")

    split_dir = os.path.join(out_root, split)
    _ensure_dir(split_dir)
    state = _scan_existing(split_dir)
    shards_dir = state["shards_dir"]

    # Load PKL (list of [2,T,H,W])
    with open(pkl_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        data = pickle.loads(mm)
        mm.close()

    if not (isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data)):
        raise ValueError("PKL must contain a list of numpy arrays, each shaped [2,T,H,W].")

    # Determine starting global index and next shard number
    append_mode = state["existing_shards"] or state["has_meta"]
    if append_mode:
        # We append to existing dataset; ignore provided start_index
        global_idx = int(state["global_next_index"])
        next_shard_num = int(state["next_shard_num"])
        print(f"[Append] Found existing split '{split}'. Starting global index at {global_idx}, next shard #{next_shard_num:05d}.")
    else:
        global_idx = int(start_index)
        next_shard_num = 0
        print(f"[New] Creating split '{split}'. Starting global index at {global_idx}.")

    total_new = len(data)
    num_new_shards = math.ceil(total_new / shard_size)
    print(f"Converting {total_new} volumes into {num_new_shards} shard(s) under {shards_dir} "
          f"(compression={compression}, pctl={pctl})")

    shard_index = state["meta"]["shards"][:] if (state["has_meta"] and "shards" in state["meta"]) else []

    # Write shards
    for s in range(num_new_shards):
        lo = s * shard_size
        hi = min(total_new, (s + 1) * shard_size)
        items = data[lo:hi]

        # Find an unused shard filename (robust if gaps exist)
        shard_num = next_shard_num
        while True:
            shard_name = f"shard_{shard_num:05d}.h5"
            shard_path = os.path.join(shards_dir, shard_name)
            if not os.path.exists(shard_path):
                break
            shard_num += 1
        next_shard_num = shard_num + 1

        _write_shard(
            shard_path, items, global_idx,
            compression=compression, pctl=pctl, norm_method="mag_percentile"
        )
        shard_index.append({"file": shard_name, "start_idx": global_idx, "count": len(items)})
        global_idx += len(items)
        print(f"  wrote {shard_name}: {len(items)} items")

    # Update / create meta
    meta = state["meta"] if state["has_meta"] else {
        "total": 0,
        "num_shards": 0,
        "shard_size": shard_size,
        "compression": compression,
        "first_global_index": start_index,
        "norm": {"method": "mag_percentile", "pctl": pctl},
        "shards": []
    }

    # Warn on differing compression / norm settings (safe to mix, but FYI)
    if state["has_meta"]:
        if meta.get("compression") != compression:
            print(f"[WARN] Existing meta compression='{meta.get('compression')}' != new '{compression}'. Shards may mix compression.")
        if meta.get("norm", {}).get("pctl") != pctl:
            print(f"[WARN] Existing meta pctl={meta.get('norm', {}).get('pctl')} != new pctl={pctl}. Shards may mix scaling percentiles.")

    meta["total"] = int((meta.get("total", 0)) + total_new)
    meta["num_shards"] = int((meta.get("num_shards", 0)) + num_new_shards)
    meta["shard_size"] = shard_size  # informational
    meta["compression"] = compression
    meta["norm"] = {"method": "mag_percentile", "pctl": pctl}
    meta["shards"] = shard_index

    with open(state["meta_path"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Meta updated at {state['meta_path']}")

def main():
    ap = argparse.ArgumentParser(description="Preprocess PKL (list of [2,T,H,W] raw) into HDF5 shards with robust normalization to [-1,1]. Append-safe.")
    ap.add_argument("--pkl", required=True, help="Path to input .pkl (list of numpy arrays [2,T,H,W])")
    ap.add_argument("--out_root", required=True, help="Output dataset root (e.g., /storage/CINE_data)")
    ap.add_argument("--split", required=True, choices=["train","val","test"], help="Subfolder to write (train/val/test)")
    ap.add_argument("--shard_size", type=int, default=256, help="#volumes per .h5 shard")
    ap.add_argument("--compression", choices=["lzf","gzip","none"], default="lzf", help="HDF5 compression")
    ap.add_argument("--start_index", type=int, default=0, help="Global start index (ignored if appending to existing split)")
    ap.add_argument("--pctl", type=float, default=99.5, help="Magnitude percentile used for scaling to ~1.0")
    args = ap.parse_args()
    preprocess(args.pkl, args.out_root, args.split, args.shard_size, args.compression, args.start_index, args.pctl)

if __name__ == "__main__":
    main()
