#!/usr/bin/env python3
"""
Download the BreakHis breast-cancer histopathology dataset, filter to a single
magnification, and write a PATIENT-DISJOINT index/query split.

Unlike NCT-CRC-HE-100K, BreakHis encodes a patient (slide) identifier in every
filename, so we can build an honest patient-disjoint evaluation: no patient that
appears in the gallery (index) set appears in the query set. This is the whole
point of adding this dataset.

Filename convention (e.g. SOB_B_TA-14-4659-200-001.png):
    <PROC>_<CLASS>_<TYPE>-<YEAR>-<SLIDE>-<MAG>-<SEQ>.png
      PROC  = biopsy procedure (SOB)
      CLASS = B (benign) | M (malignant)
      TYPE  = subtype code: A,F,PT,TA (benign) / DC,LC,MC,PC (malignant)
      YEAR-SLIDE = patient/slide identifier (we use "<YEAR>-<SLIDE>")
      MAG   = magnification (40|100|200|400)

Usage:
    python scripts/download_breakhis.py [--data-dir /data] [--magnification 200]
                                        [--index-frac 0.6] [--granularity subtype]

Source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
License: CC BY 4.0 (research use)
"""
import argparse
import json
import random
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_settings

ARCHIVE_URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
ARCHIVE_NAME = "BreaKHis_v1.tar.gz"

# Subtype code -> human-readable name (8-class "subtype" granularity)
SUBTYPES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes_tumor", "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma", "LC": "lobular_carcinoma", "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

# Parses the BreakHis filename; SLIDE may contain letters, so use \w+.
FNAME_RE = re.compile(
    r"^(?P<proc>[A-Z]+)_(?P<cls>[BM])_(?P<type>[A-Z]+)-"
    r"(?P<year>\d+)-(?P<slide>\w+)-(?P<mag>\d+)-(?P<seq>\d+)\.png$"
)


def download_archive(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Resumable, integrity-checked download. Safe to interrupt and re-run."""
    if dest.exists() and tarfile.is_tarfile(dest):
        print(f"  Already downloaded and valid: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    resume_at = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}

    print(f"  Downloading {dest.name} (resume from {resume_at} bytes) ...")
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        # 206 = partial (resume accepted); 200 = full (server ignored Range)
        if resume_at and r.status_code == 200:
            resume_at = 0  # server restarted from scratch
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_at
        mode = "ab" if resume_at else "wb"
        with open(part, mode) as f, tqdm(
            total=total, initial=resume_at, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    if not tarfile.is_tarfile(part):
        sys.exit(f"  Downloaded file is not a valid tar archive: {part}\n"
                 f"  Delete it and retry, or download manually to {dest}.")
    part.rename(dest)
    print(f"  Verified archive: {dest.name}")


def extract_archive(archive: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  Already extracted: {dest_dir}")
        return
    print(f"  Extracting {archive.name} -> {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir, filter="data")  # filter: avoid path-traversal, silence warning


def collect_records(raw_dir: Path, magnification: int, granularity: str) -> list[dict]:
    """Walk the extracted tree, keep only the chosen magnification, parse each file."""
    records = []
    skipped = 0
    for png in raw_dir.rglob("*.png"):
        m = FNAME_RE.match(png.name)
        if not m or int(m.group("mag")) != magnification:
            continue
        type_code = m.group("type")
        if type_code not in SUBTYPES:
            skipped += 1
            continue
        patient_id = f"{m.group('year')}-{m.group('slide')}"
        if granularity == "binary":
            label = "malignant" if m.group("cls") == "M" else "benign"
        else:  # subtype (8-class)
            label = type_code
        records.append({
            "path": str(png.resolve()),
            "label": label,
            "patient_id": patient_id,
            "source": "BreakHis",
        })
    if skipped:
        print(f"  WARNING: skipped {skipped} files with unrecognized subtype codes")
    return records


def patient_disjoint_split(
    records: list[dict], index_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split PATIENTS (not images) per class so gallery and query share no patient."""
    rng = random.Random(seed)

    # patient_id -> label (1:1 in BreakHis: one tumor type per slide/patient)
    patient_label = {r["patient_id"]: r["label"] for r in records}
    by_label: dict[str, list[str]] = defaultdict(list)
    for pid, label in patient_label.items():
        by_label[label].append(pid)

    index_patients: set[str] = set()
    for label, patients in by_label.items():
        patients = sorted(patients)
        rng.shuffle(patients)
        n = len(patients)
        # At least 1 patient on each side when a class has >= 2 patients.
        n_index = max(1, round(index_frac * n))
        if n >= 2:
            n_index = min(n_index, n - 1)
        index_patients.update(patients[:n_index])

    index_records = [r for r in records if r["patient_id"] in index_patients]
    query_records = [r for r in records if r["patient_id"] not in index_patients]
    return index_records, query_records


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override base data directory")
    parser.add_argument("--magnification", type=int, default=None,
                        choices=[40, 100, 200, 400])
    parser.add_argument("--index-frac", type=float, default=None,
                        help="Fraction of patients per class used as the gallery/index")
    parser.add_argument("--granularity", default=None, choices=["subtype", "binary"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # config.yaml is the source of truth; CLI flags override when provided.
    settings = load_settings(Path(args.config))
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    magnification = args.magnification or settings.breakhis.magnification
    index_frac = args.index_frac if args.index_frac is not None else settings.breakhis.index_patient_frac
    granularity = args.granularity or settings.breakhis.granularity

    # BreakHis lives under <data-dir>/breakhis to coexist with NCT-CRC.
    base = data_dir / "breakhis"
    downloads_dir = base / "downloads"
    raw_dir = base / "raw"

    # 1. Download (resumable + verified)
    download_archive(ARCHIVE_URL, downloads_dir / ARCHIVE_NAME)

    # 2. Extract
    extract_archive(downloads_dir / ARCHIVE_NAME, raw_dir)

    # 3. Collect records at the chosen magnification
    records = collect_records(raw_dir, magnification, granularity)
    if not records:
        sys.exit(f"No images found at {magnification}X under {raw_dir}")
    n_patients = len({r["patient_id"] for r in records})
    print(f"  {len(records)} images / {n_patients} patients at {magnification}X "
          f"({granularity})")

    # 4. Patient-disjoint split
    index_records, query_records = patient_disjoint_split(
        records, index_frac, args.seed
    )
    idx_pat = {r["patient_id"] for r in index_records}
    qry_pat = {r["patient_id"] for r in query_records}
    overlap = idx_pat & qry_pat
    assert not overlap, f"BUG: split is not patient-disjoint, overlap={overlap}"

    # Warn about classes that ended up with no queries (e.g. a subtype with 1 patient):
    index_labels = {r["label"] for r in index_records}
    query_labels = {r["label"] for r in query_records}
    for missing in sorted(index_labels - query_labels):
        print(f"  WARNING: class '{missing}' has no query patients "
              f"(too few patients to split) — it won't appear in eval metrics.")

    # 5. Write manifests
    write_manifest(index_records, base / "index_manifest.jsonl")
    write_manifest(query_records, base / "query_manifest.jsonl")

    print(f"\nDone (patient-disjoint).")
    print(f"  Index : {len(index_records):5d} images / {len(idx_pat)} patients")
    print(f"  Query : {len(query_records):5d} images / {len(qry_pat)} patients")
    print(f"  Shared patients between splits: {len(overlap)}  (must be 0)")
    print(f"  Manifests: {base}/index_manifest.jsonl, {base}/query_manifest.jsonl")
    print(f"\nNext: set 'dataset: breakhis' in config.yaml, then 'make build-index' and 'make eval'.")


if __name__ == "__main__":
    main()
