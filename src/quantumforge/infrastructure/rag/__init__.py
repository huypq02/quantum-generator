"""RAG (Retrieval-Augmented Generation) - Document retrieval and embeddings."""

from .retriever import load_retriever
from .factory import EmbeddingModel

__all__ = [
    "load_retriever",
    "EmbeddingModel",
]
