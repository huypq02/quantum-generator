from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IReranker(ABC):
    """Interface for document reranking."""
    @abstractmethod
    def __init__(self, config: RetrieverConfig):
        """
        Initialize the reranker adapter.

        :param config: Retriever configuration containing reranker settings.
        :type config: RetrieverConfig
        """
    
    @abstractmethod
    def rank(self):
        """
        Build a configured reranker instance.
        """
        pass
