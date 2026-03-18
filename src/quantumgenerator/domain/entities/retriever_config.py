from dataclasses import dataclass, field
from typing import Any, Dict, Sequence


@dataclass
class RetrieverConfig:
    """
    Configuration entity for document retrieval.
    
    :param retriever_type: Type of retriever to use.
    :type retriever_type: str
    :param vectordb_path: Path to the vector database.
    :type vectordb_path: str
    :param documents: Sequence of documents to retrieve from.
    :type documents: Sequence[Any]
    :param embedder: Embedding model instance.
    :type embedder: Any
    :param search_type: Type of search algorithm (default: 'mmr' for Maximal Marginal Relevance).
    :type search_type: str
    :param search_kwargs: Additional search parameters.
    :type search_kwargs: Dict[str, Any]
    """
    retriever_type: str
    vectordb_path: str
    documents: Sequence[Any]
    embedder: Any
    search_type: str = "mmr"  # Using Maximal Marginal Relevance algorithm
    search_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "k": 1,            # Top 1 most relevant document
            "lambda_mult": 0.7 # 70% focus on relevance, 30% on diversity
        }
    )
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 5
    rerank_device: str | None = None
