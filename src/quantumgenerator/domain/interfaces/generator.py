from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IGenerator(ABC):
    """Interface for text generation models."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        :param model_name: Name or path of the model.
        :type model_name: str
        :param kwargs: Additional configuration parameters.
        """
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model from external source.
        
        :return: None
        :rtype: None
        """
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
        """
        Generate text from a prompt.
        
        :param prompt: Input prompt for text generation.
        :type prompt: str
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param top_p: Nucleus sampling parameter.
        :type top_p: float
        :param kwargs: Additional generation parameters.
        :return: Generated text.
        :rtype: str
        """
        pass

    def tokenize(self, text: str) -> list:
        """
        Tokenize input text (optional).
        
        :param text: Input text to tokenize.
        :type text: str
        :return: List of tokens.
        :rtype: list
        """
        pass
