from abc import ABC, abstractmethod
from src.quantumforge.domain.entities.retriever_config import RetrieverConfig


class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, config: RetrieverConfig):
        pass
