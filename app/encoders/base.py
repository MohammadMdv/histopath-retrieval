from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class Encoder(ABC):
    name: str
    embed_dim: int
    gated: bool = False

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Return (N, D) float32 array of L2-normalized embeddings."""
        ...

    def encode_single(self, image: Image.Image) -> np.ndarray:
        return self.encode([image])[0]
