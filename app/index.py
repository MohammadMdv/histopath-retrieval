"""
FAISS index build, save, and load. Metadata stored as a JSON-lines file.
Index files are namespaced by encoder name to prevent dimension mismatches.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _faiss_path(index_dir: Path, encoder_name: str) -> Path:
    return index_dir / f"{encoder_name}.faiss"


def _meta_path(index_dir: Path, encoder_name: str) -> Path:
    return index_dir / f"{encoder_name}_meta.jsonl"


def build_index(
    embeddings: np.ndarray,  # (N, D) float32, already L2-normalized
    metadata: list[dict],    # parallel list: {id, path, label, source}
    encoder_name: str,
    index_dir: Path,
) -> faiss.IndexFlatIP:
    index_dir.mkdir(parents=True, exist_ok=True)
    assert embeddings.shape[0] == len(metadata), "embeddings and metadata must be parallel"

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(_faiss_path(index_dir, encoder_name)))
    with open(_meta_path(index_dir, encoder_name), "w") as f:
        for record in metadata:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved index: {index.ntotal} vectors, dim={dim}, encoder={encoder_name}")
    return index


def load_index(
    encoder_name: str,
    embed_dim: int,
    index_dir: Path,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    fp = _faiss_path(index_dir, encoder_name)
    mp = _meta_path(index_dir, encoder_name)

    if not fp.exists() or not mp.exists():
        raise FileNotFoundError(
            f"No index found for encoder '{encoder_name}' at {index_dir}. "
            "Run 'make build-index' first."
        )

    index = faiss.read_index(str(fp))

    # Guard: refuse to load a mismatched index
    if index.d != embed_dim:
        raise ValueError(
            f"Index dimension {index.d} does not match encoder '{encoder_name}' "
            f"embed_dim={embed_dim}. Rebuild the index with this encoder."
        )

    with open(mp) as f:
        metadata = [json.loads(line) for line in f]

    logger.info(f"Loaded index: {index.ntotal} vectors, dim={index.d}, encoder={encoder_name}")
    return index, metadata


def index_exists(encoder_name: str, index_dir: Path) -> bool:
    return (
        _faiss_path(index_dir, encoder_name).exists()
        and _meta_path(index_dir, encoder_name).exists()
    )
