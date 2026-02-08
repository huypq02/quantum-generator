from abc import ABC, abstractmethod
from src.quantumforge.domain.entities import TrainingSession, TrainingResult


class ITrainer(ABC):
    @abstractmethod
    def train(self, session: TrainingSession) -> TrainingResult:
        """
        Execute fine-tuning training pipeline.
        
        :param session: Entities of TrainingSession.
        :return: Entities of TrainingResult.
        """
        pass
