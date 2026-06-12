#!/usr/bin/env python3
"""
Download NCT-CRC-HE-100K (index set) and CRC-VAL-HE-7K (query/eval set) from Zenodo,
extract, subsample per class for the index, and write a manifest.

Usage:
    python scripts/download_data.py [--subsample N] [--data-dir /data]

Zenodo record: https://zenodo.org/record/1214456
"""
import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "1214456"
FILES = {
    "NCT-CRC-HE-100K.zip": "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip",
    "CRC-VAL-HE-7K.zip":   "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip",
}

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  Already extracted: {dest_dir.name}")
        return
    print(f"  Extracting {zip_path.name} → {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def subsample_index_images(
    src_root: Path, dest_root: Path, n_per_class: int, seed: int = 42
) -> list[dict]:
    """Copy at most n_per_class images per class from src_root to dest_root."""
    rng = random.Random(seed)
    manifest = []

    # NCT-CRC-HE-100K unzips to src_root/NCT-CRC-HE-100K/<CLASS>/*.tif
    inner = src_root / "NCT-CRC-HE-100K"
    if not inner.exists():
        # Some zips put files directly inside
        inner = src_root

    for cls in CLASSES:
        cls_dir = inner / cls
        if not cls_dir.exists():
            print(f"  WARNING: class dir not found: {cls_dir}")
            continue
        images = sorted(cls_dir.glob("*.tif")) + sorted(cls_dir.glob("*.png"))
        sampled = rng.sample(images, min(n_per_class, len(images)))

        out_dir = dest_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sampled:
            out_path = out_dir / img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)
            manifest.append({"path": str(out_path), "label": cls, "source": "NCT-CRC-HE-100K"})

    print(f"  Subsampled {len(manifest)} index images ({n_per_class}/class)")
    return manifest


def collect_query_images(src_root: Path, dest_root: Path) -> list[dict]:
    """Collect all CRC-VAL-HE-7K images as the held-out query set."""
    manifest = []
    inner = src_root / "CRC-VAL-HE-7K"
    if not inner.exists():
        inner = src_root

    for cls in CLASSES:
        cls_dir = inner / cls
        if not cls_dir.exists():
            continue
        images = sorted(cls_dir.glob("*.tif")) + sorted(cls_dir.glob("*.png"))
        out_dir = dest_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            out_path = out_dir / img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)
            manifest.append({"path": str(out_path), "label": cls, "source": "CRC-VAL-HE-7K"})

    print(f"  Collected {len(manifest)} query images from CRC-VAL-HE-7K")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=1000, help="Images per class for index")
    parser.add_argument("--data-dir", default="/data", help="Base data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip files after extract")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    downloads_dir = data_dir / "downloads"
    raw_index_dir = data_dir / "raw_index"
    raw_query_dir = data_dir / "raw_query"
    index_images_dir = data_dir / "index_images"
    query_images_dir = data_dir / "query_images"

    # 1. Download
    for fname, url in FILES.items():
        download_file(url, downloads_dir / fname)

    # 2. Extract
    extract_zip(downloads_dir / "NCT-CRC-HE-100K.zip", raw_index_dir)
    extract_zip(downloads_dir / "CRC-VAL-HE-7K.zip", raw_query_dir)

    # 3. Subsample index images
    index_manifest = subsample_index_images(
        raw_index_dir, index_images_dir, args.subsample, args.seed
    )

    # 4. Collect query images
    query_manifest = collect_query_images(raw_query_dir, query_images_dir)

    # 5. Write manifests
    with open(data_dir / "index_manifest.jsonl", "w") as f:
        for r in index_manifest:
            f.write(json.dumps(r) + "\n")

    with open(data_dir / "query_manifest.jsonl", "w") as f:
        for r in query_manifest:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone.")
    print(f"  Index images : {len(index_manifest)} → {index_images_dir}")
    print(f"  Query images : {len(query_manifest)} → {query_images_dir}")
    print(f"  Manifests    : {data_dir}/*.jsonl")
    print(f"\nNote: You can delete {raw_index_dir} and {downloads_dir} to reclaim space.")


if __name__ == "__main__":
    main()
