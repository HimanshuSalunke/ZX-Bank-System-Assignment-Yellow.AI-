"""
ZX Bank AI — SBERT Embeddings.

GPU-accelerated dense embedding generation using Sentence-Transformers.
The model (``all-MiniLM-L6-v2``) runs on the NVIDIA RTX 4050 for fast
inference. Embeddings are 384-dimensional L2-normalised vectors.

The model is downloaded ONCE on first run, then loaded from local cache
on all subsequent runs (no network access needed).
"""

from __future__ import annotations

# ── MUST be set BEFORE importing sentence_transformers ─────────────────
# This prevents HuggingFace Hub from making ANY network requests.
# The model is loaded from the local cache (~/.cache/huggingface/).
import os as _os

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, component="embeddings")

# Module-level singleton (loaded once, reused)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model from local cache (no network access)."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info(
            "loading_embedding_model",
            model=settings.embedding_model,
            device=settings.embedding_device,
        )
        try:
            _model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
        except Exception:
            # Model not cached yet — download it once with online mode
            logger.info("model_not_cached_downloading_once")
            _os.environ["HF_HUB_OFFLINE"] = "0"
            _os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
            # Re-enable offline mode
            _os.environ["HF_HUB_OFFLINE"] = "1"
            _os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info(
            "embedding_model_loaded",
            model=settings.embedding_model,
            device=str(_model.device),
            embedding_dim=_model.get_sentence_embedding_dimension(),
        )
    return _model


def init_model() -> None:
    """Eagerly load the embedding model and warm up CUDA kernels."""
    model = _get_model()
    # Warmup: run a dummy encode to force CUDA kernel compilation
    # and GPU memory allocation. Without this, the first real query
    # pays a ~1s penalty for JIT compilation.
    model.encode("warmup", normalize_embeddings=True, convert_to_numpy=True)
    logger.info("embedding_model_warmed_up")


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors."""
    return _get_model().get_sentence_embedding_dimension()


def encode_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Args:
        texts: Texts to encode.
        batch_size: Batch size for GPU inference.
        show_progress: Whether to show a progress bar.
        normalize: L2-normalise the output vectors.

    Returns:
        2-D numpy array of shape ``(len(texts), embedding_dim)``.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    logger.info(
        "texts_encoded",
        count=len(texts),
        shape=embeddings.shape,
    )
    return embeddings


def encode_query(query: str, normalize: bool = True) -> np.ndarray:
    """
    Encode a single query string into a dense vector.

    Returns:
        1-D numpy array of shape ``(embedding_dim,)``.
    """
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embedding
