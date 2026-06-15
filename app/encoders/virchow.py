"""
paige-ai/Virchow and Virchow2 — GATED on HuggingFace, requires access request.

These load as plain timm VisionTransformers: `model(x)` returns the full token
sequence `[B, N, 1280]` (NOT a DINOv2-style dict, and it takes no `is_training`).
Paige's recommended embedding is concat(class_token, mean(patch_tokens)) = 2560-d.

Register tokens differ by version and MUST be skipped before mean-pooling the
patch tokens, or the embedding is silently corrupted:
  - Virchow  (v1): 1 class token + 256 patch tokens, 0 register tokens → patch[:, 1:]
  - Virchow2     : 1 class token + 4 register tokens + 256 patch → patch[:, 5:]
"""
import numpy as np
import torch
from PIL import Image

from .base import Encoder


class VirchowEncoder(Encoder):
    name = "virchow"
    embed_dim = 2560          # concat(class_token[1280], mean_patch_tokens[1280])
    gated = True
    _hf_repo = "paige-ai/Virchow"
    _num_register_tokens = 0  # Virchow v1 has no register tokens

    def __init__(self, device: str, model_cache: str, hf_token: str):
        import timm
        from torchvision import transforms  # noqa: F401  (kept for parity / future use)

        self.model = timm.create_model(
            f"hf-hub:{self._hf_repo}", pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model.eval().to(device)
        self.device = device
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        images = self._apply_stain(images)
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
            output = self.model(batch)                 # [B, N, 1280] token sequence
        class_token = output[:, 0]                     # [B, 1280]
        patch_tokens = output[:, 1 + self._num_register_tokens:]  # [B, P, 1280]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)  # [B, 2560]
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        return embedding.cpu().numpy()


class Virchow2Encoder(VirchowEncoder):
    name = "virchow2"
    embed_dim = 2560
    gated = True
    _hf_repo = "paige-ai/Virchow2"
    _num_register_tokens = 4   # Virchow2 has 4 register tokens (skip output[:, 1:5])
