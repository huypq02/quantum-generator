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
            prompt: str, 
            max_new_tokens: int = 512, 
            temperature: float = 0.7, 
            top_p: float = 0.9,
            **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass

    def tokenize(self, text: str) -> list:
        """Tokenize input text (optional)."""
        pass