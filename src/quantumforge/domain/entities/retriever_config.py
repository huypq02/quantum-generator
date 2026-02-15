from dataclasses import dataclass, field
from typing import Any, Dict, Sequence


@dataclass
class RetrieverConfig:
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
