# Project Handoff: Histopathology Patch Retrieval

This document is written for a fresh AI assistant (or human developer) picking up this
project cold. It describes what was built, every design decision and the reasoning behind
it, the exact state of the code, known limitations, and concrete next steps.

---

## What the system does

A content-based image retrieval (CBIR) demo for histopathology. A user uploads a single
H&E-stained tissue patch; the system encodes it with a pretrained pathology foundation
model, searches a prebuilt FAISS index of patch embeddings, and returns the top-K most
similar patches with their tissue class label and cosine similarity score. A majority-vote
predicted class across the K results is shown as a convenience output.

Scope is deliberately narrow: **patch → patch only**. No WSI ingestion, no tiling, no
bag-level reasoning (Yottixel-style mosaic/median-of-minimum is explicitly out of scope).
The system is an offline research demo, not a clinical tool.

---

## Repository

- **GitHub**: https://github.com/MohammadMdv/histopath-retrieval
- **Branch**: `main`
- **Local path on the build server**: `/home/user01/histopath-retrieval/`
- **Storage**: single local disk. Large files (model weights, index, data) live inside the
  repo folder under `./model_cache`, `./index_store`, `./data` (all gitignored). The earlier
  `/mnt/vdb/histopath/` external mount is no longer used.

---

## File tree (annotated)

```
histopath-retrieval/
├── config.yaml              ← single source of truth for all runtime settings
├── .env.example             ← template for HF_TOKEN and optional DEVICE override
├── .gitignore               ← excludes data/, index_store/, model_cache/, *.faiss, .env
├── requirements.txt         ← pinned deps (Python 3.12, PyTorch cu121, FAISS-cpu, FastAPI…)
├── Dockerfile               ← nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04 base; deps via uv
├── docker-compose.yml       ← runtime: nvidia + 3 local volume mounts (model_cache, index_store, data)
├── Makefile                 ← targets: download-data, build-index, run, eval, build, stop
├── README.md                ← user-facing quickstart + license table + security notice
├── HANDOFF.md               ← this document
│
├── app/
│   ├── __init__.py
│   ├── config.py            ← loads config.yaml + .env → typed Settings (pydantic)
│   ├── main.py              ← FastAPI app: lifespan, /health, /search, /thumb/{id}, static
│   ├── index.py             ← FAISS IndexFlatIP build/save/load + metadata; encoder-tag guard
│   ├── retrieval.py         ← encode query → FAISS search → top-K + majority vote
│   └── encoders/
│       ├── __init__.py      ← re-exports Encoder, load_encoder
│       ├── base.py          ← Encoder ABC: encode(list[PIL]) → (N,D) float32 L2-normalized
│       ├── registry.py      ← name→factory map, gated detection, graceful fallback
│       ├── phikon.py        ← phikon-v2 (default) and phikon-v1
│       ├── hibou.py         ← hibou-b and hibou-l
│       ├── uni.py           ← UNI and UNI2-h (gated)
│       └── virchow.py       ← Virchow and Virchow2 (gated)
│
├── scripts/
│   ├── download_data.py     ← NCT-CRC: Zenodo fetch + extract + per-class subsample + manifests
│   ├── download_breakhis.py ← BreakHis: fetch + 200X filter + PATIENT-DISJOINT split + manifests
│   ├── build_index.py       ← batched embedding pass → FAISS + metadata JSONL (dataset-aware)
│   └── eval.py              ← Recall@K + majority-vote acc@K with bootstrap CI + patient-disjoint check
│
└── app/static/
    ├── index.html           ← single-page drag-drop upload + results grid
    ├── style.css
    └── app.js               ← fetch /health on load, POST /search, render grid
```

---

## Configuration (`config.yaml`)

```yaml
encoder: phikon-v2          # which feature extractor to use
top_k: 5                    # results returned per query
subsample_per_class: 1000   # patches per class in the index (9 classes → 9000 total)
batch_size: 64              # embedding batch size during index build
device: auto                # auto | cuda | cpu

paths:
  data_dir: /data
  index_dir: /index_store
  model_cache: /model_cache

eval:
  n_bootstrap: 1000
  query_sample: null         # null = all CRC-VAL-HE-7K queries
```

**Environment variables** (in `.env`, never committed):
- `HF_TOKEN` — HuggingFace token, required only for gated encoders
- `DEVICE` — optional override for `device` field

The `Settings` pydantic model in `app/config.py` resolves `device: auto` to `cuda` or
`cpu` at runtime by checking `torch.cuda.is_available()`. There is a minor bug on line 63:
`os.environ["DEVICE"]` should be `os.environ.get("DEVICE")` (the key checked is `"device"`
but the read uses `"DEVICE"`). Safe to fix if DEVICE env override is needed.

---

## Encoder system

### Interface (`app/encoders/base.py`)

```python
class Encoder(ABC):
    name: str         # matches config.yaml `encoder:` key
    embed_dim: int    # output dimension after L2 normalization
    gated: bool       # True = requires HF_TOKEN + access approval

    def encode(self, images: list[PIL.Image]) -> np.ndarray:
        # Returns (N, D) float32, L2-normalized
```

All encoders normalize their output so that **cosine similarity == inner product**, which
is why FAISS `IndexFlatIP` is used (not `IndexFlatL2`).

### Available encoders

| Config name | Class | HF repo | Dim | Gated | License | Notes |
|---|---|---|---|---|---|---|
| `phikon-v2` **(default)** | `PhikonV2Encoder` | `owkin/phikon-v2` | 1024 | No | Research/non-commercial | Standard `AutoModel`, no `trust_remote_code` |
| `phikon` | `PhikonV1Encoder` | `owkin/phikon` | 768 | No | Research/non-commercial | Inherits PhikonV2 |
| `hibou-b` | `HibouBEncoder` | `histai/hibou-b` | 768 | No | Apache-2.0 | Needs `trust_remote_code=True` |
| `hibou-l` | `HibouLEncoder` | `histai/hibou-L` | 1024 | No | Apache-2.0 | Needs `trust_remote_code=True` |
| `uni` | `UNIEncoder` | `MahmoodLab/UNI` | 1024 | **Yes** | Non-commercial research | timm loader, downloads `pytorch_model.bin` |
| `uni2-h` | `UNI2Encoder` | `MahmoodLab/UNI2-h` | 1536 | **Yes** | Non-commercial research | timm `hf-hub:` loader |
| `virchow` | `VirchowEncoder` | `paige-ai/Virchow` | 1280 | **Yes** | Non-commercial research | timm, CLS-only; CLS+mean-patch (2560) commented out |
| `virchow2` | `Virchow2Encoder` | `paige-ai/Virchow2` | 1280 | **Yes** | Non-commercial research | Inherits Virchow |

### Registry and fallback (`app/encoders/registry.py`)

`load_encoder(name, device, model_cache, hf_token)` looks up the encoder in `_REGISTRY`,
checks if it is gated and whether a token is present, and falls back to `phikon-v2` with
a logged warning if:
1. A gated encoder is requested but `HF_TOKEN` is absent/empty, OR
2. The encoder's `__init__` raises any exception (network failure, bad token, etc.)

The active encoder name is always reported in `/health` and shown in the page header, so
users can see which encoder is actually running.

### Adding a new encoder

1. Create `app/encoders/myencoder.py` implementing the `Encoder` ABC.
2. Add an entry to `_REGISTRY` in `app/encoders/registry.py`.
3. If gated, set `gated = True` and document the access request URL in README.
4. After switching encoders, **always rebuild the index** — the index filename is
   `<encoder_name>.faiss` and the loader guards against dimension mismatches.

---

## Index system (`app/index.py`)

- **Type**: `faiss.IndexFlatIP` — exact inner-product search (= exact cosine on L2-normalized vectors). No quantization, no IVF, no HNSW. At 9K–100K vectors this is sub-millisecond; approximate search is not needed for this scale.
- **Files on disk**:
  - `<index_dir>/<encoder_name>.faiss` — the FAISS binary
  - `<index_dir>/<encoder_name>_meta.jsonl` — one JSON record per vector: `{id, path, label, source}`
- **Dimension guard**: `load_index` checks `index.d == embed_dim` and raises `ValueError` with a clear message if they don't match (e.g., if someone swaps encoders without rebuilding).
- **IDs**: each record's `"id"` field is its 0-based position in the FAISS index. The `/thumb/{id}` endpoint uses this to look up the file path.

---

## Data pipeline

### Dataset: NCT-CRC-HE-100K + CRC-VAL-HE-7K

- **Source**: Zenodo record 1214456 (https://zenodo.org/record/1214456)
- **License**: CC BY 4.0
- **Content**: Colorectal cancer H&E patches, 224×224 pixels, 9 tissue classes:
  ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
- **Split used**: NCT-CRC-HE-100K → index; CRC-VAL-HE-7K → held-out queries/eval
- **File extension**: `.tif` (PIL handles natively)

### `scripts/download_data.py`

1. Downloads both zip files from Zenodo into `/data/downloads/` (idempotent: skips if exists).
2. Extracts to `/data/raw_index/` and `/data/raw_query/`.
3. Subsamples `--subsample N` (default 1000) images per class from the index set into
   `/data/index_images/<CLASS>/`.
4. Copies all CRC-VAL-HE-7K images to `/data/query_images/<CLASS>/`.
5. Writes `/data/index_manifest.jsonl` and `/data/query_manifest.jsonl`.

**Resumable**: zip extraction checks if the dest dir is non-empty before extracting.
Subsample copy checks if the file exists before copying.

**Space note**: `/data/raw_index/` (~7.5 GB) and `/data/downloads/` (~8 GB) can be
deleted after the index is built. Only `/data/index_images/` (~200 MB) needs to persist
for thumbnail serving.

### `scripts/build_index.py`

Reads `index_manifest.jsonl`, batches images through the encoder (batch size from config),
collects embeddings, assigns sequential `id` fields, calls `app.index.build_index()`.
Skips unreadable images with a warning rather than crashing.

---

## Web application (`app/main.py`)

FastAPI with a lifespan context manager that:
1. Loads the encoder on startup (triggers model download if not cached).
2. Loads the FAISS index if it exists; if not, the app starts but `/search` returns 503.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `app/static/index.html` |
| `GET` | `/health` | JSON: encoder name, embed_dim, index_size, device, top_k |
| `POST` | `/search` | Multipart image upload → JSON results + majority_vote |
| `GET` | `/thumb/{id}` | Returns indexed image as JPEG (converts from TIFF on-the-fly) |
| `GET` | `/static/*` | Serves CSS/JS |

### `/search` response shape

```json
{
  "results": [
    {"rank": 1, "id": 4231, "label": "TUM", "score": 0.9821, "thumb_url": "/thumb/4231"},
    ...
  ],
  "majority_vote": "TUM"
}
```

### Frontend (`app/static/`)

Vanilla JS, no framework. `app.js`:
- Calls `/health` on page load to populate the status bar (active encoder, index size, K).
- Handles drag-and-drop and file-picker upload.
- POSTs to `/search`, renders the result grid with thumbnails, labels, and scores.
- Shows the majority-vote prediction in a badge above the grid.

---

## Evaluation (`scripts/eval.py`)

Runs over CRC-VAL-HE-7K queries against the built index. Reports:

- **Recall@K**: fraction of queries where at least one of the top-K retrieved patches shares the true class label.
- **Majority-vote accuracy@K**: fraction of queries where the most common label in top-K equals the true class.
- Both reported overall and per-class, with **mean ± std and 95% bootstrap CI** (1000 bootstrap resamples by default). **No bare point estimates** — this was an explicit design requirement to avoid over-optimistic reporting.

### Patient-level holdout: NCT-CRC vs BreakHis

There are now two datasets, selected by `dataset:` in `config.yaml`:

**`nct-crc` (default)** — NCT-CRC-HE-100K does not expose per-patient identifiers in its
public Zenodo release. The train (100K) vs. val (7K) split is the only leakage control;
same-source patient exclusion is **not possible** because patient IDs are unavailable.
Metrics are upper bounds. `eval.py` detects the absence of `patient_id` and prints this
limitation.

**`breakhis` (patient-disjoint)** — BreakHis encodes a patient/slide ID in every filename.
`scripts/download_breakhis.py` parses it, keeps a single magnification (default 200X), and
writes a **stratified patient-disjoint split**: patients are split *per class* into gallery
vs. query so that no patient appears in both. `eval.py` then verifies disjointness at
runtime (computes `index_patients ∩ query_patients`) and prints `Patient-disjoint: YES/NO`.
This is the genuine patient-held-out estimate NCT-CRC can't give.

**How it's wired (dataset-agnostic infra):**
- `Settings.dataset_dir` → `data/` for NCT-CRC (back-compat), `data/breakhis/` otherwise.
- `Settings.index_tag` → bare encoder name for NCT-CRC, `breakhis__<encoder>` otherwise, so
  the two FAISS indices coexist in `index_store/`.
- Manifests gained an optional `patient_id` field; `build_index.py` passes it straight into
  the FAISS metadata JSONL, and `eval.py`'s `report_leakage_status()` reads it back.
- `app/main.py` loads `settings.index_tag` and reports `dataset` in `/health`.

Switching `dataset` requires a `make build-index` (different index file). A subtype with
only one patient (rare at 200X) lands entirely in the gallery and is skipped in queries;
the download script warns when this happens.

### Class imbalance: voting strategies + macro metrics

BreakHis's gallery is ~45% ductal carcinoma, which lets plain majority vote win minority
neighborhoods and tanks their accuracy (the Recall≫Accuracy gap seen in the first run).
Two additions address this, both **query-time only — no index rebuild**:

- **`voting` config key** (`uniform` | `distance` | `inverse_freq` | `distance_invfreq`),
  implemented in `app/retrieval.py::vote()`. `inverse_freq` weights each neighbor by
  `gallery_class_count(label)^(-vote_beta)` (see `gallery_class_weights(metadata, beta)`);
  `distance_invfreq` also multiplies by cosine score. The **`vote_beta`** config key tempers
  the correction (0=off, 1=full 1/count, ~0.5=soft inverse-sqrt-freq); full 1/count
  over-corrects on BreakHis (one rare neighbor outvotes several correct ones), so β≈0.5 is the
  practical optimum. Used by both `eval.py` and the live app (`/search`), so the web badge and
  the metrics agree. `uniform` is the default → NCT-CRC behavior unchanged.
- **Macro-averaged metrics** in `eval.py::macro_bootstrap_ci()`: the run now prints micro
  (per-query) *and* macro (per-class-averaged) Recall/Accuracy with bootstrap CIs. Macro is
  the honest headline for the imbalanced 8-class task.

---

## Docker and deployment

### `Dockerfile`

- Base: `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04` (matches the host: Ubuntu 24.04,
  driver 580 / CUDA 13 capable, TITAN RTX 24 GB)
- Python 3.12 installed via apt (Ubuntu 24.04 default)
- **Dependencies installed with `uv`, not pip** — the `uv` static binary is copied from
  `ghcr.io/astral-sh/uv:latest`. Relevant env vars: `UV_SYSTEM_PYTHON=1` (install into the
  system interpreter), `UV_BREAK_SYSTEM_PACKAGES=1` (bypass PEP 668 on Ubuntu 24.04),
  `UV_INDEX_STRATEGY=unsafe-best-match` (lets uv pull `torch` from the PyTorch index and
  everything else from PyPI, like pip does), `UV_NO_CACHE=1`.
- The base image ships an NVIDIA apt source that 403s without a proxy; the Dockerfile
  removes `/etc/apt/sources.list.d/cuda*.list` and `nvidia*.list` before `apt-get update`.
  CUDA/cuDNN runtime libs are already baked into the image layers, so this is harmless.
- PyTorch cu121 wheels from `https://download.pytorch.org/whl/cu121` (forward-compatible
  with the host's newer CUDA driver)
- No data, no weights, no index baked into the image — all in volumes
- Works CPU-only: remove or comment out the `runtime: nvidia` line in `docker-compose.yml`

### `docker-compose.yml`

Three local volume mounts (inside the repo folder, all gitignored):

| Host | Container | Contents |
|---|---|---|
| `./model_cache` | `/model_cache` | HF/timm weights |
| `./index_store` | `/index_store` | FAISS index + metadata JSONL |
| `./data` | `/data` | Dataset images + manifests |

GPU access uses `runtime: nvidia` plus `NVIDIA_VISIBLE_DEVICES=all` /
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` env vars. (The old
`deploy.resources.reservations.devices` block was replaced because this host's
nvidia-container-toolkit is configured in CDI mode and rejected that path — it explicitly
asks for `--runtime=nvidia`.) **Requires `nvidia-container-toolkit`** on the host.
The obsolete top-level `version:` key was removed (Compose v2+ ignores it and warns).

`config.yaml` is also bind-mounted read-only so encoder/K changes don't require a rebuild.

### `Makefile`

```
make build          # docker compose build
make download-data  # runs download_data.py inside the container
make build-index    # runs build_index.py inside the container
make run            # docker compose up (foreground)
make eval           # runs eval.py inside the container
make stop           # docker compose down
```

`SUBSAMPLE_PER_CLASS` env var overrides the default 1000 for `download-data`.

---

## Expected resource footprint (default config)

| Item | Size |
|---|---|
| phikon-v2 weights (cached) | ~1.2 GB |
| FAISS index (9,000 × 1024 float32) | ~37 MB |
| Metadata JSONL | < 1 MB |
| Index images (224² TIFF, 9K files) | ~200 MB |
| CRC-VAL-HE-7K query images | ~350 MB |
| Raw download (can delete after build) | ~15 GB |
| **Peak VRAM during index build** (batch=64, fp16 autocast) | **< 3 GB** |
| **Peak VRAM during query inference** | **< 2 GB** |
| Build/eval GPU (current host) | NVIDIA TITAN RTX 24 GB |

---

## Known issues and TODOs

### Bugs / rough edges

1. **`config.py` — DEVICE env var bug**: ✅ FIXED. The guard now checks `"DEVICE"` (was
   `"device"`) to match the `os.environ["DEVICE"]` read, so the `DEVICE` override works.

2. **`registry.py` lines 51–54 — duplicate branch**: Both `if needs_token` and `else`
   execute identical code (the `hf_token` kwarg is always passed). The conditional is
   harmless but dead code. Safe to simplify to one line.

3. **`download_data.py` — `--keep-zips` flag is parsed but not acted on**: The flag is
   defined with `argparse` but there is no `if not args.keep_zips: shutil.rmtree(...)`.
   Either wire it up or remove the flag.

4. **`download_data.py` — unused imports**: `hashlib` and `io` are imported but never
   used. Remove them.

5. **Thumbnail serving re-encodes on every request**: `/thumb/{id}` opens the TIFF from
   disk and converts to JPEG on every call. For a demo this is fine, but at higher load
   a small thumbnail cache (e.g. `functools.lru_cache` keyed by image_id) would help.

### Missing features (not in scope for MVP but natural next steps)

- **Authentication**: No auth at all. Fine for local use; add HTTP Basic Auth or a
  simple token header before exposing on any network.
- **ANN index support**: `IndexFlatIP` is exact but O(N) per query. At 100K+ vectors,
  adding `IndexIVFFlat` or `IndexHNSWFlat` would speed up queries. The `app/index.py`
  module is clean enough to add this behind a config flag.
- **Batch query upload**: The frontend handles one image at a time. A batch mode (zip
  upload → JSON array response) would be useful for running eval without the scripts.
- **UMAP / t-SNE visualization**: Embedding a 2D projection of the index alongside the
  query position would help users understand retrieval quality intuitively.
- **Patient-level leakage exclusion in eval**: If a patient-ID mapping for NCT-CRC
  becomes available, add a `--exclude-same-source` flag to `eval.py` that filters
  retrieved patches from the same patient as the query.
- **Progress endpoint for index build**: Currently `build_index.py` is a fire-and-forget
  script. A `/build-status` SSE endpoint would let the UI show progress.

---

## How to resume development

### First-time setup on the existing server

```bash
cd /home/user01/histopath-retrieval
cp .env.example .env           # fill in HF_TOKEN if using a gated encoder
make build                     # 5–10 min
make download-data             # downloads ~15 GB from Zenodo, then subsamples
make build-index               # ~10 min on V100
make run                       # visit http://localhost:8000
```

### To switch encoders

1. Edit `config.yaml`: change `encoder: phikon-v2` to e.g. `encoder: hibou-b`
2. If gated: set `HF_TOKEN` in `.env`
3. `make build-index` (the old index is kept; the new one is written as a separate file)
4. `make run`

### To extend the codebase

- **New encoder**: add `app/encoders/myencoder.py` + entry in `registry.py`
- **Different dataset**: replace `download_data.py` with a script that produces the same
  `index_manifest.jsonl` format: one JSON record per line with `{path, label, source}`
- **Different index type**: edit `app/index.py` — `build_index` and `load_index` are
  the only two functions to change
- **API changes**: `app/main.py` is straightforward FastAPI; add endpoints there
- **Frontend changes**: `app/static/` is plain HTML/CSS/JS, no build step needed

---

## Decisions that were made and why (do not revisit without reason)

| Decision | Rationale |
|---|---|
| Default encoder: `phikon-v2` | Ungated (runs out-of-the-box), standard `AutoModel` loader (no `trust_remote_code`), 1024-dim CLS, well-documented benchmark results |
| Cosine via `IndexFlatIP` on L2-normalized vectors | Exact, correct, zero ambiguity about normalization; avoids `IndexFlatL2` semantics confusion |
| Index namespaced by encoder name | Prevents silent garbage results if encoder is switched without rebuild |
| Subsample default 1000/class (9K total) | ~37 MB index, ~200 MB thumbnails, ~3 min GPU embed time — comfortable on 22 GB free disk |
| Bootstrap CI in eval, not bare point estimates | Explicit requirement from the domain; the histopath literature has a known problem with over-optimistic single-number evals |
| Patient-leakage documented as limitation for NCT-CRC | NCT-CRC Zenodo release has no patient IDs; fabricating exclusions would be worse than acknowledging the gap |
| BreakHis added for genuine patient-disjoint eval | BreakHis encodes a patient/slide ID per filename, enabling a real patient-held-out split. Default 200X + 8 subtypes; both datasets coexist (namespaced data dir + index tag) so leaky-vs-disjoint can be compared |
| faiss-cpu not faiss-gpu | At 9K–100K vectors exact CPU search is sub-millisecond; faiss-gpu adds CUDA dependency friction and is unnecessary at this scale |
| No JS framework (vanilla) | MVP; no build step; easier for others to read and extend |
| Workers=1 in uvicorn CMD | The encoder and FAISS index are global state loaded once at startup; multiple workers would each load their own copy (doubling VRAM), which is not worth it for a single-user demo |
