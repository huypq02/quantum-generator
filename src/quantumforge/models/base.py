from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        """Initialize the model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.config = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """Load model from external source."""
        pass