"""
MahmoodLab/UNI and UNI2-h — GATED on HuggingFace, requires access request.
Loaded via timm with local model weights downloaded through huggingface_hub.
"""
import numpy as np
import torch
from PIL import Image

from .base import Encoder


class UNIEncoder(Encoder):
    name = "uni"
    embed_dim = 1024
    gated = True

    def __init__(self, device: str, model_cache: str, hf_token: str):
        import timm
        from torchvision import transforms
        from huggingface_hub import hf_hub_download

        local_dir = f"{model_cache}/MahmoodLab/UNI/assets/ckpts"
        import os; os.makedirs(local_dir, exist_ok=True)
        hf_hub_download(
            "MahmoodLab/UNI", filename="pytorch_model.bin",
            local_dir=local_dir, token=hf_token
        )
        self.model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16,
            init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.model.load_state_dict(
            torch.load(f"{local_dir}/pytorch_model.bin", map_location="cpu")
        )
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
            embeddings = self.model(batch)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return embeddings.cpu().numpy()


class UNI2Encoder(UNIEncoder):
    name = "uni2-h"
    embed_dim = 1536
    gated = True

    def __init__(self, device: str, model_cache: str, hf_token: str):
        import timm
        from torchvision import transforms
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download("MahmoodLab/UNI2-h", cache_dir=model_cache, token=hf_token)
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True,
            init_values=1e-5, dynamic_img_size=True
        )
        self.model.eval().to(device)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
