from abc import ABC, abstractmethod

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self):
        pass
