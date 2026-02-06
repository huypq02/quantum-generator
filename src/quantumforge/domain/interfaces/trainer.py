from abc import ABC, abstractmethod
from src.quantumforge.domain.entities import TrainingSession


class ITrainer(ABC):
    @abstractmethod
    def train(self, session: TrainingSession):
        """
        Execute fine-tuning training pipeline.
        
        :param session: Entities of TrainingSession.
        """
        pass
