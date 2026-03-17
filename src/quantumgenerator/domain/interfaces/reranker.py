from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IReranker(ABC):
    """Interface for document reranking."""

    @abstractmethod
    def rank(self, kwargs):
        """
        Rerank documents by relevance to query.
        """
        pass
