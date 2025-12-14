from .deepseek import DeepSeekModel
from .codegemma import CodeGemmaModel
from .qwen import QwenModel

class ModelFactory:
    @staticmethod
    def create_model(self, model_type: str, **kwargs):
        """Create a specific model."""
        if model_type == "deepseek":
            return DeepSeekModel(**kwargs)
        elif model_type == "codegemma":
            return CodeGemmaModel(**kwargs)
        elif model_type == "qwen":
            return QwenModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")