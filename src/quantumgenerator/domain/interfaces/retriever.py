from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IRetriever(ABC):
    """Interface for document retrieval."""

    @abstractmethod
    def retrieve(self, query: str, config: RetrieverConfig):
        """
        Retrieve relevant documents based on query.
        
        :param query: User query string.
        :type query: str
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Retrieved documents.
        """
        pass
