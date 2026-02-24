from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IReranker(ABC):
    """Interface for document reranking."""

    @abstractmethod
    def rank(self, query: str, config: RetrieverConfig):
        """
        Rerank documents by relevance to query.
        
        :param query: User query string.
        :type query: str
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Reranked documents.
        """
        pass
