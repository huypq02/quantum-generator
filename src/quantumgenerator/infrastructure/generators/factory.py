from .deepseek_generator import DeepSeekModel
from .codegemma_generator import CodeGemmaModel
from .qwen_generator import QwenModel
from .codellama_generator import CodeLlamaModel

class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_model(model_type: str, **kwargs):
        """
        Create a specific model instance.
        
        :param model_type: Type of model to create (e.g., 'deepseek', 'codegemma', 'qwen', 'codellama').
        :type model_type: str
        :param kwargs: Additional configuration parameters for the model.
        :return: Model instance.
        :raises ValueError: If model type is unknown.
        """
        if model_type == "deepseek":
            return DeepSeekModel(**kwargs)
        elif model_type == "codegemma":
            return CodeGemmaModel(**kwargs)
        elif model_type == "qwen":
            return QwenModel(**kwargs)
        elif model_type == "codellama":
            return CodeLlamaModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
