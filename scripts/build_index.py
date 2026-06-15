#!/usr/bin/env python3
"""
Batch-embed index_images and build the FAISS index.

Usage:
    python scripts/build_index.py [--config config.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_settings
from app.encoders import load_encoder
from app.index import build_index
from app.augment import build_augmentation_plan, apply_augmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    settings = load_settings(Path(args.config))
    device = settings.resolved_device
    print(f"Device  : {device}")
    print(f"Dataset : {settings.dataset}")
    print(f"Encoder : {settings.encoder}")
    print(f"Stain   : {settings.stain_norm}")

    encoder = load_encoder(
        settings.encoder, device, str(settings.model_cache), settings.hf_token,
        stain_norm=settings.stain_norm,
    )
    # Guard against the registry's silent fallback. Building an index is a
    # benchmarking action: if the requested encoder failed to load and we
    # fell back to a different one, the index would be saved under the
    # requested name but hold the wrong model's vectors. Abort loudly instead.
    if encoder.name != settings.encoder:
        sys.exit(
            f"ERROR: requested encoder '{settings.encoder}' but loaded "
            f"'{encoder.name}' (fallback). Refusing to build a mislabeled index "
            f"'{settings.index_tag}'.\n"
            f"Fix the cause (gated model needs HF_TOKEN, or a load error above) "
            f"or set encoder: {encoder.name} in config.yaml to build it on purpose."
        )
    print(f"Embed dim: {encoder.embed_dim}")

    manifest_path = settings.dataset_dir / "index_manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\n"
                 f"Run the download step for dataset '{settings.dataset}' first.")

    with open(manifest_path) as f:
        records = [json.loads(l) for l in f]

    if settings.augment.enabled:
        records, aug_summary = build_augmentation_plan(
            records,
            settings.augment.target_per_class,
            settings.augment.max_factor,
            settings.augment.seed,
        )
        print(f"Augment : ON (target_per_class={settings.augment.target_per_class}, "
              f"max_factor={settings.augment.max_factor})")
        for label, (orig, added) in sorted(aug_summary.items()):
            if added:
                print(f"  {label:<6} {orig:4d} -> {orig + added:4d}  (+{added} augmented)")
    else:
        print("Augment : off")

    print(f"Images to embed: {len(records)}")

    all_embeddings = []
    valid_records = []
    batch_size = settings.batch_size

    for i in tqdm(range(0, len(records), batch_size), desc="Embedding"):
        batch_records = records[i : i + batch_size]
        images = []
        batch_valid = []
        for r in batch_records:
            try:
                img = Image.open(r["path"]).convert("RGB")
                if "augment" in r:
                    img = apply_augmentation(img, r["augment"])
                images.append(img)
                batch_valid.append(r)
            except Exception as e:
                print(f"  Skipping {r['path']}: {e}")

        if not images:
            continue

        all_embeddings.append(encoder.encode(images))
        valid_records.extend(batch_valid)

    embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {embeddings_np.shape}")

    for idx, r in enumerate(valid_records):
        r["id"] = idx

    build_index(
        embeddings_np,
        valid_records,
        settings.index_tag,
        settings.index_dir,
    )
    print(f"Index '{settings.index_tag}' saved to {settings.index_dir}")


if __name__ == "__main__":
    main()
