"""
Encoder registry with gated-model detection and graceful fallback to phikon-v2.
"""
import logging
from typing import Optional

from .base import Encoder

logger = logging.getLogger(__name__)

_DEFAULT = "phikon-v2"

# Maps config name → (factory, requires_token)
_REGISTRY: dict[str, tuple] = {
    "phikon-v2": ("app.encoders.phikon", "PhikonV2Encoder", False),
    "phikon":    ("app.encoders.phikon", "PhikonV1Encoder", False),
    "hibou-b":   ("app.encoders.hibou",  "HibouBEncoder",   False),
    "hibou-l":   ("app.encoders.hibou",  "HibouLEncoder",   False),
    "uni":       ("app.encoders.uni",    "UNIEncoder",       True),
    "uni2-h":    ("app.encoders.uni",    "UNI2Encoder",      True),
    "virchow":   ("app.encoders.virchow","VirchowEncoder",   True),
    "virchow2":  ("app.encoders.virchow","Virchow2Encoder",  True),
}


def load_encoder(
    name: str,
    device: str,
    model_cache: str,
    hf_token: Optional[str] = None,
    stain_norm: str = "none",
) -> Encoder:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(_REGISTRY)}")

    module_path, cls_name, needs_token = _REGISTRY[name]

    if needs_token and not hf_token:
        logger.warning(
            f"Encoder '{name}' is gated and requires HF_TOKEN. "
            f"Falling back to default '{_DEFAULT}'. "
            f"To use '{name}', request access on HuggingFace and set HF_TOKEN in .env."
        )
        name = _DEFAULT
        module_path, cls_name, needs_token = _REGISTRY[name]

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    try:
        if needs_token:
            encoder = cls(device=device, model_cache=model_cache, hf_token=hf_token)
        else:
            encoder = cls(device=device, model_cache=model_cache, hf_token=hf_token)
    except Exception as exc:
        if name != _DEFAULT:
            logger.warning(
                f"Failed to load encoder '{name}': {exc}. "
                f"Falling back to '{_DEFAULT}'."
            )
            return load_encoder(_DEFAULT, device, model_cache, hf_token, stain_norm)
        raise

    from app.preprocess import make_stain_normalizer
    encoder.stain_normalizer = make_stain_normalizer(stain_norm)

    logger.info(
        f"Loaded encoder '{encoder.name}' (dim={encoder.embed_dim}, device={device}, "
        f"stain_norm={stain_norm})"
    )
    return encoder
