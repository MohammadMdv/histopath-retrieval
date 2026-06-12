# Histopathology Patch Retrieval — Demo MVP

> **What this does:** You upload a single H&E-stained tissue patch; the system returns the top-K most visually similar patches from a prebuilt index of colorectal cancer patches (NCT-CRC-HE-100K), each with its tissue class and cosine similarity score.
>
> **What this does NOT do:** It is not a diagnostic tool. It performs patch-level image→image retrieval only. No WSI-level analysis. No patient-level reasoning. Not clinically validated. Not hardened for public internet exposure.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose v2 | `docker compose version` |
| **nvidia-container-toolkit** | Required for GPU. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| ~25 GB free disk on the volume mount | 8 GB Zenodo download + model cache + index |
| NVIDIA GPU (recommended) | V100 16 GB or similar; CPU-only works, but index build is much slower |

---

## Quickstart (default: phikon-v2, 1000 patches/class)

```bash
git clone <this-repo> && cd histopath-retrieval
cp .env.example .env          # leave HF_TOKEN blank for the default encoder

make build                    # build Docker image (~5 min first time)
make download-data            # ~8 GB download from Zenodo, then subsample
make build-index              # embed 9000 patches, build FAISS index (~10 min GPU)
make run                      # start at http://localhost:8000
```

Open `http://localhost:8000`, drag-and-drop a 224×224 H&E patch (e.g. from `CRC-VAL-HE-7K`), and see the top-5 most similar indexed patches.

To stop: `make stop`

---

## Expected Disk & VRAM Footprint (default config)

| Item | Size |
|---|---|
| phikon-v2 weights (cached) | ~1.2 GB |
| FAISS index (9,000 × 1024-dim) | ~37 MB |
| Subsampled index images | ~200 MB |
| CRC-VAL-HE-7K query images | ~350 MB |
| NCT-CRC-HE-100K download (can delete after build) | ~7–8 GB |
| **Peak VRAM during index build** (batch=64, fp16) | **< 3 GB** |
| **Peak VRAM during query inference** | **< 2 GB** |

---

## Switching Encoders

Edit `config.yaml`:

```yaml
encoder: hibou-b    # Apache-2.0 license, 768-dim, ungated
```

Available encoders:

| Name | Dim | Gated | License |
|---|---|---|---|
| `phikon-v2` **(default)** | 1024 | No | Research/non-commercial |
| `phikon` | 768 | No | Research/non-commercial |
| `hibou-b` | 768 | No | Apache-2.0 |
| `hibou-l` | 1024 | No | Apache-2.0 |
| `uni` | 1024 | **Yes** | See below |
| `uni2-h` | 1536 | **Yes** | See below |
| `virchow` | 1280 | **Yes** | See below |
| `virchow2` | 1280 | **Yes** | See below |

After switching encoders, you **must rebuild the index** (`make build-index`). The app guards against loading a mismatched index.

---

## Using Gated Models (UNI, UNI2-h, Virchow, Virchow2)

1. Request access on HuggingFace for the relevant model:
   - UNI / UNI2-h: https://huggingface.co/MahmoodLab/UNI
   - Virchow / Virchow2: https://huggingface.co/paige-ai/Virchow
2. Generate a HuggingFace token: https://huggingface.co/settings/tokens
3. Add to `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxx
   ```
4. Set `encoder: uni` (or other) in `config.yaml`.
5. Run `make build-index` (model will be downloaded on first run).

If the token is missing or access is not granted, the app falls back to `phikon-v2` automatically with a warning in the logs.

---

## Offline Evaluation

```bash
make eval
```

Reports **Recall@K** and **majority-vote accuracy@K** over held-out CRC-VAL-HE-7K queries, with mean ± std and 95% bootstrap CI. Results are printed per-class and overall.

**Leakage note:** NCT-CRC-HE-100K does not expose per-patient identifiers in its public Zenodo release. The train-set (100K) vs. val-set (7K) split is the only leakage control. Patient-level same-source exclusion is not performed and is documented as a limitation. Treat reported numbers as upper bounds.

---

## Configuration Reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `encoder` | `phikon-v2` | Feature extractor (see encoder table above) |
| `top_k` | `5` | Number of retrieved results |
| `subsample_per_class` | `1000` | Patches per class for index (9 classes → 9K total) |
| `batch_size` | `64` | Embedding batch size during index build |
| `device` | `auto` | `auto` / `cuda` / `cpu` |

---

## Licenses

| Component | License |
|---|---|
| NCT-CRC-HE-100K dataset | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| phikon / phikon-v2 | Research/non-commercial ([owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2)) |
| hibou-b / hibou-l | Apache-2.0 ([histai/hibou-b](https://huggingface.co/histai/hibou-b)) |
| UNI / UNI2-h | **Gated** — request access; non-commercial research license |
| Virchow / Virchow2 | **Gated** — request access; non-commercial research license |
| FAISS | MIT |
| FastAPI / Uvicorn | MIT |

---

## Security Notice

This application has **no authentication, no HTTPS, and no rate limiting**. It is intended for local or private network use only. Do not expose it to the public internet.

---

## Volumes

Three directories are mounted from the host into the container:

| Host path | Container path | Contents |
|---|---|---|
| `/mnt/vdb/histopath/model_cache` | `/model_cache` | HuggingFace / timm model weights |
| `/mnt/vdb/histopath/index_store` | `/index_store` | FAISS index + metadata |
| `/mnt/vdb/histopath/data` | `/data` | Downloaded + subsampled patch images |

Change the host paths in `docker-compose.yml` if your storage is elsewhere.
