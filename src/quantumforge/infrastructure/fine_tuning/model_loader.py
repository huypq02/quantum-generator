from src.quantumforge.models.factory import ModelFactory
from src.quantumforge.models.deepseek import DeepSeekModel
from src.quantumforge.models.codegemma import CodeGemmaModel
from src.quantumforge.models.qwen import QwenModel
from src.quantumforge.models.codellama import CodeLlamaModel


def load_model(
        model_type: str,
        model_name: str
) -> DeepSeekModel | CodeGemmaModel | QwenModel | CodeLlamaModel:
    model = ModelFactory.create_model(
        model_type=model_type,
        model_name=model_name
    )
    model.load_model()

    return model
