"""
Embedding interface for the Geometry of Meaning project.

Provides a unified interface for embedding models, managing:
  - Model loading and caching
  - Batching and device selection
  - Normalization
  - Vector serialization
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "multilingual-e5-large": {
        "hf_name": "intfloat/multilingual-e5-large",
        "version": "1.0.0",
        "dimensionality": "1024",
        "description": "Multilingual E5 large model from Microsoft",
    },
    "bge-m3": {
        "hf_name": "BAAI/bge-m3",
        "version": "1.0.0",
        "dimensionality": "1024",
        "description": "BGE M3 multilingual embedding model from BAAI",
    },
}


def get_model_info(model_id: str) -> dict[str, str]:
    """
    Get metadata for a registered embedding model.

    Args:
        model_id: The model identifier (e.g., 'multilingual-e5-large').

    Returns:
        A dict with model metadata (hf_name, version, dimensionality, description).

    Raises:
        ValueError: If the model_id is not in the registry.
    """
    if model_id not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return dict(_MODEL_REGISTRY[model_id])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model_id: str = "multilingual-e5-large",
    batch_size: int = 32,
    normalize: bool = True,
    device: str | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts using the specified model.

    Args:
        texts: List of text strings to embed.
        model_id: Model identifier from the registry.
        batch_size: Number of texts to process per batch.
        normalize: Whether to L2-normalize the output embeddings.
        device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
        use_cache: Whether to cache embeddings to avoid recomputation.

    Returns:
        A numpy array of shape (len(texts), embedding_dim) with the embeddings.

    Raises:
        ValueError: If model_id is unknown or texts is empty.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install it with: pip install sentence-transformers torch"
        ) from err

    model = _load_model(model_id, device, use_cache)

    # E5 models use "passage:" prefix for symmetric comparison tasks.
    if "e5" in model_id.lower():
        texts = [f"passage: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return np.array(embeddings)


# Module-level model cache
_MODEL_CACHE: dict[str, Any] = {}


def _load_model(
    model_id: str,
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    """
    Load and cache an embedding model.

    Args:
        model_id: Model identifier from the registry.
        device: Device string. Auto-detected if None.
        use_cache: Whether to cache and reuse loaded models.

    Returns:
        A SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    if use_cache and model_id in _MODEL_CACHE:
        logger.debug(f"Returning cached model: {model_id}")
        return _MODEL_CACHE[model_id]

    model_info = get_model_info(model_id)
    logger.info(f"Loading model: {model_info['hf_name']}")

    model = SentenceTransformer(model_info["hf_name"], device=device)

    if use_cache:
        _MODEL_CACHE[model_id] = model

    return model


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings so each row has unit norm.

    Args:
        embeddings: Array of shape (n, d).

    Returns:
        L2-normalized array of the same shape.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
    return embeddings / norms
