"""Acquire, validate and unify the corn leaf disease dataset.

The widely used Kaggle "Corn or Maize Leaf Disease Dataset" (Ghose, 2020) is a curated combination of
PlantVillage corn images and PlantDoc corn images. This module rebuilds an equivalent 4-class dataset
directly from the two original, licensed sources (no Kaggle account needed):

  PlantVillage (Hughes & Salathé 2015; Mohanty et al. 2016), CC BY-SA 3.0, github.com/spMohanty/PlantVillage-Dataset
  PlantDoc (Singh et al. 2020, CODS-COMAD), CC BY 4.0, github.com/pratikkayal/PlantDoc-Dataset

Class mapping
  Blight          <- PV "Northern_Leaf_Blight"                  + PlantDoc "Corn leaf blight"
  Common_Rust     <- PV "Common_rust_"                          + PlantDoc "Corn rust leaf"
  Gray_Leaf_Spot  <- PV "Cercospora_leaf_spot Gray_leaf_spot"   + PlantDoc "Corn Gray leaf spot"
  Healthy         <- PV "healthy"

Leakage control: PlantVillage contains several photos of the same physical leaf (and "copy" duplicates).
Its published leaf-grouping map assigns every photo to a leaf group, and splits are made by group.
Exact duplicates (identical decoded pixels) are also removed across both sources.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.dedup import dihedral_hashes, near_duplicate_groups
from src.utils.common import get_logger

log = get_logger(__name__)

PV_REPO = "https://github.com/spMohanty/PlantVillage-Dataset"
PD_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset"
PV_MAP = {
    "Corn_(maize)___Northern_Leaf_Blight": "Blight",
    "Corn_(maize)___Common_rust_": "Common_Rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray_Leaf_Spot",
    "Corn_(maize)___healthy": "Healthy",
}
PD_MAP = {"Corn leaf blight": "Blight", "Corn rust leaf": "Common_Rust", "Corn Gray leaf spot": "Gray_Leaf_Spot"}


def _git(*args, cwd=None, input_=None):
    return subprocess.run(["git", *args], cwd=cwd, input=input_, check=True, capture_output=True)


def download_sources(raw_dir: Path) -> None:
    """Partial, sparse clones: only corn images (~150 MB) are transferred, not the multi-GB repositories."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pv = raw_dir / "plantvillage"
    if not (pv / "raw" / "color").exists():
        _git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", PV_REPO, str(pv))
        _git("sparse-checkout", "init", "--no-cone", cwd=pv)
        _git("sparse-checkout", "set", "/raw/color/Corn_(maize)___*/", "/leaf_grouping/leaf-map.json", cwd=pv)
        _git("checkout", "master", cwd=pv)
    pd_dir = raw_dir / "plantdoc_corn"
    if not pd_dir.exists():
        _extract_plantdoc(raw_dir / "plantdoc_repo", pd_dir)


def _extract_plantdoc(repo: Path, out: Path) -> None:
    """PlantDoc has file names containing '?' etc. that cannot exist on Windows, so blobs are read from the
    git object store and written with sanitised names instead of being checked out."""
    if not repo.exists():
        _git("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", PD_REPO, str(repo))
    listing = _git("ls-tree", "-r", "-z", "HEAD", "--", "train", "test", cwd=repo).stdout.decode().split("\0")
    entries = []
    for line in filter(None, listing):
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        split, cls = path.split("/")[:2]
        if typ == "blob" and cls in PD_MAP:
            entries.append((sha, path))
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", "--no-tags", "--filter=blob:none", "--stdin"],
                   input=("\n".join(s for s, _ in entries) + "\n").encode(), capture_output=True)
    for sha, path in entries:
        split, cls, name = path.split("/", 2)
        dest = out / split / cls / re.sub(r'[<>:"/\\|?*%&=]', "_", name)[:120]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_git("cat-file", "blob", sha, cwd=repo).stdout)
    log.info("Extracted %d PlantDoc corn images", len(entries))


def _pv_original_id(fname: str) -> str:
    stem = Path(fname).stem.split("___", 1)[-1]
    return re.sub(r"(\s+copy(\s+\d+)?)+$", "", stem).strip()


def _load_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)          # handles non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def square_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize the short side to `size` then centre-crop to size x size (keeps aspect ratio)."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def build_manifest(raw_dir: Path, processed_dir: Path, image_size: int, min_side: int) -> pd.DataFrame:
    """Validate every image with OpenCV, resize to image_size, de-duplicate, and write a manifest CSV."""
    leaf_map = json.loads((raw_dir / "plantvillage" / "leaf_grouping" / "leaf-map.json").read_text())
    candidates = []
    for folder, cls in PV_MAP.items():
        for p in sorted((raw_dir / "plantvillage" / "raw" / "color" / folder).glob("*")):
            oid = _pv_original_id(p.name)
            entry = leaf_map.get(oid)
            group = f"pv:{entry[0].split(':::')[1]}" if entry else f"pv-img:{oid}"
            candidates.append((p, cls, "PlantVillage", group, entry is not None))
    for split in ("train", "test"):
        for folder, cls in PD_MAP.items():
            for p in sorted((raw_dir / "plantdoc_corn" / split / folder).glob("*")):
                candidates.append((p, cls, "PlantDoc", f"pd:{split}/{p.name}", False))

    img_dir = processed_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    rows, seen, rejected, hash_imgs = [], {}, [], []
    for p, cls, src, group, in_map in candidates:
        img = _load_rgb(p)
        if img is None:
            rejected.append((str(p), "unreadable"))
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            rejected.append((str(p), f"too small {w}x{h}"))
            continue
        small = square_resize(img, image_size)
        digest = hashlib.md5(small.tobytes()).hexdigest()
        if digest in seen:
            if seen[digest] != cls:
                rejected.append((str(p), "duplicate with conflicting label"))
            else:
                rejected.append((str(p), "exact duplicate"))
            continue
        seen[digest] = cls
        out = img_dir / cls / f"{digest}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(out))
        rows.append({"path": str(out.relative_to(processed_dir)).replace("\\", "/"), "label": cls, "source": src,
                     "group": group, "leaf_map_hit": in_map, "orig_width": w, "orig_height": h,
                     "orig_file": p.name})
        hash_imgs.append(cv2.resize(small, (64, 64), interpolation=cv2.INTER_AREA))
    man = pd.DataFrame(rows)
    man["name_group"] = man["group"]
    # Merge near-duplicates (same photo re-saved, rotated or flipped) into one group, per class.
    for cls, idx in man.groupby("label").groups.items():
        hashes = np.stack([dihedral_hashes(cv2.cvtColor(np.asarray(hash_imgs[i]), cv2.COLOR_RGB2GRAY)) for i in idx])
        comp = near_duplicate_groups(hashes, initial_groups=man.loc[idx, "name_group"].tolist())
        man.loc[idx, "group"] = [f"{cls}:{c}" for c in comp]
    log.info("Groups: %d by name, %d after near-duplicate merging", man.name_group.nunique(), man.group.nunique())
    man.to_csv(processed_dir / "manifest.csv", index=False)
    pd.DataFrame(rejected, columns=["file", "reason"]).to_csv(processed_dir / "rejected.csv", index=False)
    log.info("Kept %d images, rejected %d", len(man), len(rejected))
    return man
