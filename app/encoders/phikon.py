"""
owkin/phikon-v2  — ViT-L/16, 1024-dim CLS embedding, ungated, research license.
Standard transformers AutoModel — no trust_remote_code needed.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .base import Encoder


class PhikonV2Encoder(Encoder):
    name = "phikon-v2"
    embed_dim = 1024
    gated = False
    _hf_repo = "owkin/phikon-v2"

    def __init__(self, device: str, model_cache: str, hf_token=None):
        self.device = device
        cache_dir = str(model_cache)
        self.processor = AutoImageProcessor.from_pretrained(
            self._hf_repo, cache_dir=cache_dir, token=hf_token
        )
        self.model = AutoModel.from_pretrained(
            self._hf_repo, cache_dir=cache_dir, token=hf_token
        )
        self.model.eval().to(device)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu().numpy()


class PhikonV1Encoder(PhikonV2Encoder):
    name = "phikon"
    embed_dim = 768
    _hf_repo = "owkin/phikon"
