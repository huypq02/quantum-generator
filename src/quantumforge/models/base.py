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

    @abstractmethod
    def generate(
            self, 
            prompt, 
            max_new_tokens = 512, 
            temperature = 0.7, 
            top_p = 0.95, 
            **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass

    def tokenize(self, text: str) -> list:
        """Tokenize input text (optional)."""
        pass