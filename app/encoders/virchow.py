"""
paige-ai/Virchow and Virchow2 — GATED on HuggingFace, requires access request.
Embedding: CLS + mean of patch tokens concatenated → 2560-dim; or CLS-only → 1280-dim.
Default here: CLS-only (1280) for simplicity and speed.
"""
import numpy as np
import torch
from PIL import Image

from .base import Encoder


class VirchowEncoder(Encoder):
    name = "virchow"
    embed_dim = 1280
    gated = True
    _hf_repo = "paige-ai/Virchow"

    def __init__(self, device: str, model_cache: str, hf_token: str):
        import timm
        from torchvision import transforms

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
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
            output = self.model(batch, is_training=True)
        # CLS token only (1280-dim); change to CLS+mean-patch (2560) by uncommenting below
        cls_token = output["x_norm_clstoken"]
        # patch_tokens = output["x_norm_patchtokens"]
        # cls_token = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        embeddings = torch.nn.functional.normalize(cls_token.float(), dim=-1)
        return embeddings.cpu().numpy()


class Virchow2Encoder(VirchowEncoder):
    name = "virchow2"
    embed_dim = 1280
    gated = True
    _hf_repo = "paige-ai/Virchow2"
