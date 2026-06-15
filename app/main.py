"""
FastAPI application: /health, /search, /thumb/{id}, static frontend.
"""
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import load_settings
from .encoders import load_encoder
from .index import load_index, index_exists
from .retrieval import search, gallery_class_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = load_settings()
_encoder = None
_index = None
_metadata: list[dict] = []
_class_weights: dict[str, float] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _encoder, _index, _metadata, _class_weights

    device = settings.resolved_device
    logger.info(f"Starting up — device={device}, encoder={settings.encoder}")

    _encoder = load_encoder(
        settings.encoder, device, str(settings.model_cache), settings.hf_token,
        stain_norm=settings.stain_norm,
    )

    if index_exists(settings.index_tag, settings.index_dir):
        _index, _metadata = load_index(settings.index_tag, _encoder.embed_dim, settings.index_dir)
        _class_weights = gallery_class_weights(_metadata, settings.vote_beta)
    else:
        logger.warning(
            "No index found — /search will return an error until you run 'make build-index'."
        )

    yield

    _encoder = None
    _index = None
    _metadata = []
    _class_weights = {}


app = FastAPI(title="Histopathology Patch Retrieval", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dataset": settings.dataset,
        "encoder": _encoder.name if _encoder else None,
        "embed_dim": _encoder.embed_dim if _encoder else None,
        "index_size": _index.ntotal if _index else 0,
        "device": settings.resolved_device,
        "top_k": settings.top_k,
        "voting": settings.voting,
        "stain_norm": settings.stain_norm,
    }


@app.post("/search")
async def search_endpoint(file: UploadFile = File(...)):
    if _encoder is None:
        raise HTTPException(503, "Encoder not loaded")
    if _index is None:
        raise HTTPException(503, "Index not built. Run 'make build-index' first.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    result = search(image, _encoder, _index, _metadata, top_k=settings.top_k,
                    voting=settings.voting, class_weights=_class_weights)
    return JSONResponse(result)


@app.get("/thumb/{image_id}")
async def thumbnail(image_id: int):
    if not _metadata:
        raise HTTPException(503, "Index not loaded")
    if image_id < 0 or image_id >= len(_metadata):
        raise HTTPException(404, "Image not found")

    path = Path(_metadata[image_id]["path"])
    if not path.exists():
        raise HTTPException(404, f"Image file missing: {path}")

    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
