from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class Encoder(ABC):
    name: str
    embed_dim: int
    gated: bool = False
    # Optional callable(PIL.Image) -> PIL.Image set by load_encoder (e.g. stain
    # normalization). Applied inside encode() so gallery and queries always go
    # through the identical preprocessing.
    stain_normalizer = None

    def _apply_stain(self, images: list[Image.Image]) -> list[Image.Image]:
        if self.stain_normalizer is None:
            return images
        return [self.stain_normalizer(img) for img in images]

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Return (N, D) float32 array of L2-normalized embeddings."""
        ...

    def encode_single(self, image: Image.Image) -> np.ndarray:
        return self.encode([image])[0]
