#!/usr/bin/env python3
"""
Offline evaluation: Recall@K and majority-vote accuracy@K over CRC-VAL-HE-7K.
Reports mean ± std and 95% bootstrap CI — no bare point estimates.

Usage:
    python scripts/eval.py [--config config.yaml] [--query-sample N]

Patient-level exclusion limitation:
    NCT-CRC-HE-100K (Zenodo release) does not expose per-patient identifiers.
    We use the train (100K index) vs. val (7K query) split as the primary
    leakage control. Same-source patient exclusion is NOT performed because
    patient IDs are unavailable. This is a known limitation: results may be
    slightly optimistic if the same patient appears in both splits (the dataset
    paper does not guarantee otherwise). Treat reported numbers as upper bounds
    on retrieval performance, not as clinical evidence.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_settings
from app.encoders import load_encoder
from app.index import load_index, index_exists
from app.retrieval import search


def bootstrap_ci(values: list[float], n: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, std, (lower, upper)) for the given values."""
    arr = np.array(values)
    mean = arr.mean()
    std = arr.std()
    boot_means = [np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(mean), float(std), (float(lo), float(hi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query-sample", type=int, default=None,
                        help="Evaluate on a random subset of queries (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = load_settings(Path(args.config))
    device = settings.resolved_device

    encoder = load_encoder(
        settings.encoder, device, str(settings.model_cache), settings.hf_token
    )

    if not index_exists(encoder.name, settings.index_dir):
        sys.exit("No index found. Run 'make build-index' first.")

    index, metadata = load_index(encoder.name, encoder.embed_dim, settings.index_dir)
    k = settings.top_k

    query_manifest_path = settings.data_dir / "query_manifest.jsonl"
    if not query_manifest_path.exists():
        sys.exit(f"Query manifest not found: {query_manifest_path}\nRun 'make download-data' first.")

    with open(query_manifest_path) as f:
        queries = [json.loads(l) for l in f]

    sample_size = args.query_sample or settings.eval.query_sample
    if sample_size:
        rng = random.Random(args.seed)
        queries = rng.sample(queries, min(sample_size, len(queries)))

    print(f"Evaluating {len(queries)} queries, top-K={k}, encoder={encoder.name}")
    print()

    recall_hits: list[float] = []
    vote_hits: list[float] = []

    # Per-class tracking
    classes = sorted(set(q["label"] for q in queries))
    class_recall: dict[str, list[float]] = {c: [] for c in classes}
    class_vote:   dict[str, list[float]] = {c: [] for c in classes}

    for q in tqdm(queries, desc="Querying"):
        try:
            img = Image.open(q["path"]).convert("RGB")
        except Exception as e:
            print(f"  Skipping {q['path']}: {e}")
            continue

        result = search(img, encoder, index, metadata, top_k=k)
        true_label = q["label"]

        retrieved_labels = [r["label"] for r in result["results"]]
        hit = float(true_label in retrieved_labels)
        vote_hit = float(result["majority_vote"] == true_label)

        recall_hits.append(hit)
        vote_hits.append(vote_hit)
        class_recall[true_label].append(hit)
        class_vote[true_label].append(vote_hit)

    n_boot = settings.eval.n_bootstrap
    np.random.seed(args.seed)

    r_mean, r_std, r_ci = bootstrap_ci(recall_hits, n_boot)
    v_mean, v_std, v_ci = bootstrap_ci(vote_hits, n_boot)

    print("=" * 60)
    print(f"OVERALL  (n={len(recall_hits)}, bootstrap n={n_boot})")
    print(f"  Recall@{k}      : {r_mean:.4f} ± {r_std:.4f}  95% CI [{r_ci[0]:.4f}, {r_ci[1]:.4f}]")
    print(f"  MajVoteAcc@{k}  : {v_mean:.4f} ± {v_std:.4f}  95% CI [{v_ci[0]:.4f}, {v_ci[1]:.4f}]")
    print()
    print(f"PER-CLASS  Recall@{k}  |  MajVoteAcc@{k}")
    print("-" * 60)
    for cls in classes:
        cr = class_recall[cls]
        cv = class_vote[cls]
        if not cr:
            continue
        rm, rs, rci = bootstrap_ci(cr, n_boot)
        vm, vs, vci = bootstrap_ci(cv, n_boot)
        print(
            f"  {cls:<6} n={len(cr):4d}  "
            f"R@K {rm:.3f}±{rs:.3f} [{rci[0]:.3f},{rci[1]:.3f}]  "
            f"Acc {vm:.3f}±{vs:.3f} [{vci[0]:.3f},{vci[1]:.3f}]"
        )

    print()
    print("LIMITATION: Patient-level same-source exclusion was NOT performed.")
    print("  NCT-CRC-HE-100K does not expose per-patient identifiers.")
    print("  The train/val split (100K index vs 7K queries) is the only leakage control.")
    print("  Reported numbers may be slightly optimistic if patients overlap across splits.")


if __name__ == "__main__":
    main()
