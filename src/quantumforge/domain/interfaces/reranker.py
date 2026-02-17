from abc import ABC, abstractmethod
from src.quantumforge.domain.entities.retriever_config import RetrieverConfig


class IReranker(ABC):
    @abstractmethod
    def rank(self, query: str, config: RetrieverConfig):
        pass
