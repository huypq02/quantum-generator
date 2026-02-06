from abc import ABC, abstractmethod


class ITrainer(ABC):
    @abstractmethod
    def train(self, dataset, config):
        """
        Execute fine-tuning training pipeline.
        
        :param dataset: Training dataset (HuggingFace Dataset, path, or file)
        :param config: Training configuration
        """
        pass
