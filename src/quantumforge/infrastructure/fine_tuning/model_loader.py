from src.quantumforge.infrastructure.generators import (
    ModelFactory,
    DeepSeekModel,
    CodeGemmaModel,
    QwenModel,
    CodeLlamaModel
)


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
