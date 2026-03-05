from abc import ABC, abstractmethod
from quantumgenerator.domain import RetrieverConfig


class RAGPipeline(ABC):
    @abstractmethod
    def get_context(self, query: str, config: RetrieverConfig) -> str:
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
