"""
histai/hibou-b and hibou-L — ViT-B/L, Apache-2.0 license, ungated.
Requires trust_remote_code=True.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .base import Encoder


class HibouEncoder(Encoder):
    gated = False

    def __init__(self, hf_repo: str, embed_dim: int, device: str, model_cache: str, hf_token=None):
        self.device = device
        cache_dir = str(model_cache)
        self.processor = AutoImageProcessor.from_pretrained(
            hf_repo, cache_dir=cache_dir, token=hf_token, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            hf_repo, cache_dir=cache_dir, token=hf_token, trust_remote_code=True
        )
        self.embed_dim = embed_dim
        self.model.eval().to(device)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        images = self._apply_stain(images)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu().numpy()


class HibouBEncoder(HibouEncoder):
    name = "hibou-b"

    def __init__(self, device: str, model_cache: str, hf_token=None):
        super().__init__("histai/hibou-b", 768, device, model_cache, hf_token)


class HibouLEncoder(HibouEncoder):
    name = "hibou-l"

    def __init__(self, device: str, model_cache: str, hf_token=None):
        super().__init__("histai/hibou-L", 1024, device, model_cache, hf_token)
