#!/usr/bin/env python3
"""
Offline evaluation: Recall@K and majority-vote accuracy@K over the query manifest
of the active dataset. Reports mean ± std and 95% bootstrap CI — no bare point
estimates.

Usage:
    python scripts/eval.py [--config config.yaml] [--query-sample N]

Patient-level holdout:
    If the manifests carry a `patient_id` (e.g. BreakHis), the run verifies and
    prints that the gallery and query patient sets are disjoint — a genuine
    patient-held-out evaluation. If they do not (e.g. NCT-CRC, which exposes no
    patient IDs), the run instead documents that only the dataset-provided
    train/val split controls leakage, so numbers should be read as upper bounds.
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
from app.retrieval import search, gallery_class_weights


def bootstrap_ci(values: list[float], n: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, std, (lower, upper)) for the given values."""
    arr = np.array(values)
    mean = arr.mean()
    std = arr.std()
    boot_means = [np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(mean), float(std), (float(lo), float(hi))


def macro_bootstrap_ci(
    class_values: dict[str, list[float]], n: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    """
    Macro (class-balanced) metric: average the per-class means, giving every class
    equal weight regardless of how many queries it has. CI by resampling within each
    class, then averaging the class means each bootstrap iteration.
    """
    arrs = {c: np.array(v) for c, v in class_values.items() if v}
    classes = list(arrs)
    point = float(np.mean([arrs[c].mean() for c in classes]))
    boot_macros = [
        float(np.mean([
            np.random.choice(arrs[c], size=len(arrs[c]), replace=True).mean()
            for c in classes
        ]))
        for _ in range(n)
    ]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_macros, [alpha, 1 - alpha])
    return point, float(np.std(boot_macros)), (float(lo), float(hi))


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

    if not index_exists(settings.index_tag, settings.index_dir):
        sys.exit("No index found. Run 'make build-index' first.")

    index, metadata = load_index(settings.index_tag, encoder.embed_dim, settings.index_dir)
    k = settings.top_k

    query_manifest_path = settings.dataset_dir / "query_manifest.jsonl"
    if not query_manifest_path.exists():
        sys.exit(f"Query manifest not found: {query_manifest_path}\n"
                 f"Run the download step for dataset '{settings.dataset}' first.")

    with open(query_manifest_path) as f:
        queries = [json.loads(l) for l in f]

    sample_size = args.query_sample or settings.eval.query_sample
    if sample_size:
        rng = random.Random(args.seed)
        queries = rng.sample(queries, min(sample_size, len(queries)))

    class_weights = gallery_class_weights(metadata, settings.vote_beta)

    print(f"Dataset : {settings.dataset}")
    print(f"Voting  : {settings.voting}"
          + (f" (beta={settings.vote_beta})" if settings.voting in ("inverse_freq", "distance_invfreq") else ""))
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

        result = search(img, encoder, index, metadata, top_k=k,
                        voting=settings.voting, class_weights=class_weights)
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
    mr_mean, mr_std, mr_ci = macro_bootstrap_ci(class_recall, n_boot)
    mv_mean, mv_std, mv_ci = macro_bootstrap_ci(class_vote, n_boot)

    print("=" * 60)
    print(f"OVERALL  (n={len(recall_hits)}, bootstrap n={n_boot})")
    print("  Micro (per-query average — dominated by large classes):")
    print(f"    Recall@{k}      : {r_mean:.4f} ± {r_std:.4f}  95% CI [{r_ci[0]:.4f}, {r_ci[1]:.4f}]")
    print(f"    MajVoteAcc@{k}  : {v_mean:.4f} ± {v_std:.4f}  95% CI [{v_ci[0]:.4f}, {v_ci[1]:.4f}]")
    print("  Macro (class-balanced — every class weighted equally):")
    print(f"    Recall@{k}      : {mr_mean:.4f} ± {mr_std:.4f}  95% CI [{mr_ci[0]:.4f}, {mr_ci[1]:.4f}]")
    print(f"    MajVoteAcc@{k}  : {mv_mean:.4f} ± {mv_std:.4f}  95% CI [{mv_ci[0]:.4f}, {mv_ci[1]:.4f}]")
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
    report_leakage_status(metadata, queries)


def report_leakage_status(metadata: list[dict], queries: list[dict]) -> None:
    """
    Verify and report whether the evaluation is patient-disjoint.

    If both the index metadata and the queries carry a `patient_id`, we can check
    directly that no patient appears in both the gallery and the query set (a true
    patient-held-out evaluation). Otherwise (e.g. NCT-CRC, which exposes no patient
    IDs) we fall back to documenting the limitation.
    """
    index_patients = {m["patient_id"] for m in metadata if "patient_id" in m}
    query_patients = {q["patient_id"] for q in queries if "patient_id" in q}

    print("=" * 60)
    if index_patients and query_patients:
        overlap = index_patients & query_patients
        disjoint = not overlap
        print("PATIENT-LEVEL HOLDOUT CHECK")
        print(f"  Index patients : {len(index_patients)}")
        print(f"  Query patients : {len(query_patients)}")
        print(f"  Shared patients: {len(overlap)}")
        print(f"  Patient-disjoint: {'YES' if disjoint else 'NO'}")
        if disjoint:
            print("  -> Every query is from a patient NOT present in the gallery.")
            print("     Metrics above are a genuine patient-held-out estimate.")
        else:
            print(f"  -> WARNING: {len(overlap)} patient(s) appear in both splits;")
            print("     metrics are optimistic. Shared:", sorted(overlap)[:10])
    else:
        print("LIMITATION: Patient-level same-source exclusion was NOT performed.")
        print("  This dataset does not expose per-patient identifiers.")
        print("  The train/val split is the only leakage control.")
        print("  Reported numbers may be optimistic if patients overlap across splits.")


if __name__ == "__main__":
    main()
