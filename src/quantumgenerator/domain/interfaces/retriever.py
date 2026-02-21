from abc import ABC, abstractmethod
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, config: RetrieverConfig):
        pass
