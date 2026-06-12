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


class Settings(BaseModel):
    encoder: str = "phikon-v2"
    top_k: int = 5
    subsample_per_class: int = 1000
    batch_size: int = 64
    device: str = "auto"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    hf_token: Optional[str] = Field(default=None)

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir)

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
    if "device" in os.environ:
        raw["device"] = os.environ["DEVICE"]

    return Settings(**raw, hf_token=hf_token)


settings = load_settings()
