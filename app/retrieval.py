"""
Encode a query image, search FAISS, return top-K results with majority vote.
"""
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from .encoders import Encoder


def search(
    query_image: Image.Image,
    encoder: Encoder,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    top_k: int = 5,
) -> dict:
    query_vec = encoder.encode([query_image.convert("RGB")])  # (1, D)
    scores, indices = index.search(query_vec, top_k)

    results = []
    labels = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        label = meta.get("label", "unknown")
        labels.append(label)
        results.append({
            "rank": rank + 1,
            "id": meta.get("id", int(idx)),
            "label": label,
            "score": float(score),
            "thumb_url": f"/thumb/{meta.get('id', int(idx))}",
        })

    majority_vote = Counter(labels).most_common(1)[0][0] if labels else "unknown"
    return {"results": results, "majority_vote": majority_vote}
