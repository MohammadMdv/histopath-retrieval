"""
Encode a query image, search FAISS, return top-K results with a (weighted) vote.

Voting strategies (config key `voting`):
  uniform          – plain majority vote (1 per neighbor). Default; matches the
                     original behavior.
  distance         – weight each neighbor by its cosine similarity score.
  inverse_freq     – weight each neighbor by count^(-beta), where count is its class
                     size in the gallery, so a dominant class can't win by sheer
                     volume. `beta` tempers the correction: 0 = none (uniform),
                     1 = full 1/count (aggressive, can over-correct), ~0.5 = a soft
                     inverse-sqrt-frequency that usually maximizes macro accuracy.
  distance_invfreq – product of `distance` and `inverse_freq`.

`inverse_freq` / `distance_invfreq` need gallery class weights; compute them once
with `gallery_class_weights(metadata, beta)` and pass them in.
"""
from collections import Counter
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from PIL import Image

from .encoders import Encoder


def gallery_class_weights(metadata: list[dict], beta: float = 1.0) -> dict[str, float]:
    """
    Tempered inverse-frequency weight per class: count^(-beta).
      beta=0 -> all weights 1.0 (no correction); beta=1 -> 1/count (full);
      beta~0.5 -> inverse-sqrt-frequency (soft correction).
    """
    counts = Counter(m.get("label", "unknown") for m in metadata)
    return {label: float(n) ** (-beta) for label, n in counts.items()}


def vote(
    labels: list[str],
    scores: list[float],
    voting: str = "uniform",
    class_weights: Optional[dict[str, float]] = None,
) -> str:
    """Aggregate neighbor labels into a single predicted class under the given strategy."""
    cw = class_weights or {}
    tally: dict[str, float] = {}
    for label, score in zip(labels, scores):
        if voting == "distance":
            w = max(float(score), 0.0)
        elif voting == "inverse_freq":
            w = cw.get(label, 1.0)
        elif voting == "distance_invfreq":
            w = max(float(score), 0.0) * cw.get(label, 1.0)
        else:  # uniform
            w = 1.0
        tally[label] = tally.get(label, 0.0) + w
    if not tally:
        return "unknown"
    return max(tally.items(), key=lambda kv: kv[1])[0]


def search(
    query_image: Image.Image,
    encoder: Encoder,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    top_k: int = 5,
    voting: str = "uniform",
    class_weights: Optional[dict[str, float]] = None,
) -> dict:
    query_vec = encoder.encode([query_image.convert("RGB")])  # (1, D)
    scores, indices = index.search(query_vec, top_k)

    results = []
    labels = []
    label_scores = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        label = meta.get("label", "unknown")
        labels.append(label)
        label_scores.append(float(score))
        results.append({
            "rank": rank + 1,
            "id": meta.get("id", int(idx)),
            "label": label,
            "score": float(score),
            "thumb_url": f"/thumb/{meta.get('id', int(idx))}",
        })

    majority_vote = vote(labels, label_scores, voting, class_weights)
    return {"results": results, "majority_vote": majority_vote}
