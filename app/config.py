import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class PathsConfig(BaseModel):
    data_dir: str = "/data"
    index_dir: str = "/index_store"
    model_cache: str = "/model_cache"


class EvalConfig(BaseModel):
    n_bootstrap: int = 1000
    query_sample: Optional[int] = None


class BreakHisConfig(BaseModel):
    magnification: int = 200          # 40 | 100 | 200 | 400
    index_patient_frac: float = 0.6   # fraction of patients per class used as the gallery
    granularity: str = "subtype"      # subtype (8-class) | binary


class AugmentConfig(BaseModel):
    enabled: bool = False             # gallery-side minority-class expansion (index build only)
    target_per_class: int = 200       # expand under-represented classes up to ~this many exemplars
    max_factor: int = 8               # cap variants per source patch (<= this many incl. original)
    seed: int = 42


class Settings(BaseModel):
    dataset: str = "nct-crc"          # nct-crc | breakhis
    encoder: str = "phikon-v2"
    top_k: int = 5
    voting: str = "uniform"           # uniform | distance | inverse_freq | distance_invfreq
    vote_beta: float = 1.0            # tempering for inverse_freq: 0=none, 1=full 1/count, ~0.5=soft
    stain_norm: str = "none"          # none | macenko  (applied symmetrically to gallery + queries)
    subsample_per_class: int = 1000
    batch_size: int = 64
    device: str = "auto"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    breakhis: BreakHisConfig = Field(default_factory=BreakHisConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    hf_token: Optional[str] = Field(default=None)

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir)

    @property
    def dataset_dir(self) -> Path:
        # NCT-CRC keeps the flat /data layout (back-compat); other datasets get a subdir.
        if self.dataset == "nct-crc":
            return self.data_dir
        return self.data_dir / self.dataset

    @property
    def index_tag(self) -> str:
        # Namespace the index by dataset so multiple datasets can coexist.
        # NCT-CRC keeps the bare encoder name for back-compat with existing index files.
        if self.dataset == "nct-crc":
            base = self.encoder
        else:
            base = f"{self.dataset}__{self.encoder}"
        # A different stain normalization produces different embeddings, so it
        # must get its own index file (and keeps the A/B from clobbering the
        # un-normalized index). stain_norm=none preserves existing tags.
        if self.stain_norm and self.stain_norm != "none":
            base = f"{base}__stain-{self.stain_norm}"
        # Augmentation changes the gallery contents, so it also gets its own
        # index file rather than overwriting the un-augmented one.
        if self.augment.enabled:
            base = f"{base}__aug-{self.augment.target_per_class}"
        return base

    @property
    def index_dir(self) -> Path:
        return Path(self.paths.index_dir)

    @property
    def model_cache(self) -> Path:
        return Path(self.paths.model_cache)

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_settings(config_path: Path = _CONFIG_PATH) -> Settings:
    raw = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    hf_token = os.environ.get("HF_TOKEN") or raw.pop("hf_token", None) or None
    if "DEVICE" in os.environ:
        raw["device"] = os.environ["DEVICE"]

    return Settings(**raw, hf_token=hf_token)


settings = load_settings()
