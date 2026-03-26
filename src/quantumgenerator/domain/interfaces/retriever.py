from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IRetriever(ABC):
    """Interface for document retrieval."""

    @abstractmethod
    def index_documents(self, config: RetrieverConfig) -> None:
        """
        Build or refresh the vector index using provided documents.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        """
        pass

    @abstractmethod
    def retrieve_context(self, config: RetrieverConfig):
        """
        Retrieve relevant documents based on query.
        
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Retrieved documents.
        """
        pass
