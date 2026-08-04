"""
Embedding interface for the Geometry of Meaning project.

Provides a unified interface for embedding models, managing:
  - Model loading and caching
  - Batching and device selection
  - Normalization
  - Vector serialization
"""

import logging
from typing import Any, Optional

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
    device: Optional[str] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts using the specified model.

    This is a stub implementation. To enable actual embedding, install the
    optional 'embeddings' dependencies and uncomment the implementation below.

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

    model_info = get_model_info(model_id)

    # TODO: Implement actual embedding using sentence-transformers
    # ------------------------------------------------------------
    # from sentence_transformers import SentenceTransformer
    #
    # model = _load_model(model_id, device)
    #
    # if "e5" in model_id.lower():
    #     texts = [f"query: {t}" for t in texts]
    #
    # embeddings = model.encode(
    #     texts,
    #     batch_size=batch_size,
    #     normalize_embeddings=normalize,
    #     show_progress_bar=True,
    # )
    # return np.array(embeddings)

    raise NotImplementedError(
        f"Embedding not yet implemented. Model '{model_id}' ({model_info['hf_name']}) "
        f"is registered but the embedding backend is a stub. "
        f"Install sentence-transformers and torch, then implement embed_texts()."
    )


def _load_model(model_id: str, device: Optional[str] = None) -> Any:
    """
    Load and cache an embedding model.

    Args:
        model_id: Model identifier from the registry.
        device: Device string. Auto-detected if None.

    Returns:
        A SentenceTransformer model instance.
    """
    # TODO: Implement model loading with caching
    # from sentence_transformers import SentenceTransformer
    #
    # model_info = get_model_info(model_id)
    # model = SentenceTransformer(model_info["hf_name"], device=device)
    # return model

    raise NotImplementedError("Model loading not yet implemented.")


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
