# Histopathology Patch Retrieval — Demo MVP

> **What this does:** You upload a single H&E-stained tissue patch; the system returns the top-K most visually similar patches from a prebuilt index of colorectal cancer patches (NCT-CRC-HE-100K), each with its tissue class and cosine similarity score.
>
> **What this does NOT do:** It is not a diagnostic tool. It performs patch-level image→image retrieval only. No WSI-level analysis. No patient-level reasoning. Not clinically validated. Not hardened for public internet exposure.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose v2+ | `docker compose version` (tested on Compose v5 / Ubuntu 24.04) |
| **nvidia-container-toolkit** | Required for GPU. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html |
| ~25 GB free disk in the project folder | 8 GB Zenodo download + model cache + index |
| NVIDIA GPU (recommended) | TITAN RTX / V100 or similar; CPU-only works, but index build is much slower |

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

Reports **Recall@K** and **majority-vote accuracy@K** over the active dataset's held-out queries, with mean ± std and 95% bootstrap CI. Results are printed per-class and overall.

**Leakage note (NCT-CRC):** NCT-CRC-HE-100K does not expose per-patient identifiers in its public Zenodo release. The train-set (100K) vs. val-set (7K) split is the only leakage control. Patient-level same-source exclusion is not performed. Treat NCT-CRC numbers as upper bounds. To get a *genuine* patient-held-out estimate, use the BreakHis dataset below.

---

## Patient-Disjoint Evaluation (BreakHis)

NCT-CRC cannot answer "how does retrieval behave when the query patient was never in the
gallery?" because it ships no patient IDs. **BreakHis** does — a patient/slide identifier is
encoded in every filename — so we can build a genuinely **patient-disjoint** split where no
patient appears in both the gallery and the query set.

```bash
# 1. Point the pipeline at BreakHis
#    edit config.yaml:  dataset: breakhis
make download-breakhis        # downloads BreakHis, filters to 200X, writes a patient-disjoint split
make build-index              # embeds the gallery patches (index namespaced as breakhis__<encoder>)
make eval                     # Recall@K / majority-vote acc@K, patient-disjoint
```

`make eval` verifies and prints the holdout status on every run, e.g.:

```
PATIENT-LEVEL HOLDOUT CHECK
  Index patients : 49
  Query patients : 33
  Shared patients: 0
  Patient-disjoint: YES
  -> Every query is from a patient NOT present in the gallery.
```

BreakHis options (in `config.yaml` under `breakhis:`):

| Key | Default | Description |
|---|---|---|
| `magnification` | `200` | Which magnification to use (40 / 100 / 200 / 400); a single value avoids a magnification confound |
| `index_patient_frac` | `0.6` | Fraction of patients **per class** placed in the gallery (the rest become queries) |
| `granularity` | `subtype` | `subtype` = 8 tumor types; `binary` = benign/malignant |

The two datasets coexist: BreakHis data lives under `data/breakhis/` and its index is named
`breakhis__<encoder>.faiss`, so switching `dataset:` back to `nct-crc` restores the original
pipeline without a re-download. **Switching `dataset` requires rebuilding the index.**

### Handling class imbalance (voting + macro metrics)

BreakHis's gallery is dominated by one class (ductal carcinoma ≈ 45% of patches), so plain
majority vote lets it win neighborhoods it shouldn't, crushing minority-class accuracy. Two
knobs address this — **no index rebuild needed**, just re-run `make eval`:

- **`voting`** (config): `inverse_freq` weights each retrieved neighbor by `count^(-vote_beta)`
  (gallery class size), and `distance_invfreq` also multiplies by similarity. Both stop a
  dominant class from winning by sheer volume. (`uniform` is the original behavior.)
  **`vote_beta`** tempers the correction: `1.0` = full `1/count` (aggressive — can let one rare
  neighbor outvote several correct ones), `~0.5` = soft inverse-sqrt-frequency (usually the best
  macro accuracy), `0` = off. Sweep it; no rebuild needed.
- **Macro metrics**: `make eval` now prints both **micro** (per-query average, dominated by
  large classes) and **macro** (class-balanced, every class weighted equally) Recall/Accuracy.
  Macro is the honest headline for this imbalanced 8-class problem.

The live web app uses the same `voting` setting for its majority-vote badge.

> BreakHis is downloaded from the official UFPR mirror (CC BY 4.0, research use). The download
> is resumable and integrity-checked, so an interrupted transfer won't corrupt the archive.

---

## Configuration Reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `dataset` | `nct-crc` | Active dataset: `nct-crc` or `breakhis` (patient-disjoint) |
| `encoder` | `phikon-v2` | Feature extractor (see encoder table above) |
| `top_k` | `5` | Number of retrieved results |
| `voting` | `uniform` | Vote aggregation: `uniform`, `distance`, `inverse_freq`, `distance_invfreq` |
| `vote_beta` | `1.0` | Tempering for `inverse_freq`: 0=off, 1=full 1/count, ~0.5=soft (sweep for best macro acc) |
| `subsample_per_class` | `1000` | NCT-CRC only: patches per class for index (9 classes → 9K total) |
| `batch_size` | `64` | Embedding batch size during index build |
| `device` | `auto` | `auto` / `cuda` / `cpu` |
| `breakhis.magnification` | `200` | BreakHis: magnification factor (40/100/200/400) |
| `breakhis.index_patient_frac` | `0.6` | BreakHis: fraction of patients/class in the gallery |
| `breakhis.granularity` | `subtype` | BreakHis: `subtype` (8-class) or `binary` |

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

Three directories are mounted from the project folder into the container (created
automatically on first run):

| Host path | Container path | Contents |
|---|---|---|
| `./model_cache` | `/model_cache` | HuggingFace / timm model weights |
| `./index_store` | `/index_store` | FAISS index + metadata |
| `./data` | `/data` | Downloaded + subsampled patch images |

All three live inside the repo directory and are gitignored. If you prefer to store the
large files elsewhere (e.g. a separate disk), change the host side of these mounts in
`docker-compose.yml`.
