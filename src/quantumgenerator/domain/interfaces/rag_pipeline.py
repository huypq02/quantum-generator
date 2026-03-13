from abc import ABC, abstractmethod
from quantumgenerator.domain import RetrieverConfig


class RAGPipeline(ABC):
    @abstractmethod
    def index_documents(self, config: RetrieverConfig) -> None:
        """
        Build or refresh the configured retrieval index.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        """
        pass

    @abstractmethod
    def retrieve_context(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context for a given query.
        
        :param query: User input query.
        :type query: str
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Retrieved context as a single string.
        :rtype: str
        """
        pass
