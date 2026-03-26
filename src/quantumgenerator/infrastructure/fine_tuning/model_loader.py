from quantumgenerator.infrastructure.generators import (
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
    """
    Load and initialize a model of the specified type.
    
    :param model_type: Type of model to load.
    :type model_type: str
    :param model_name: Name or path of the model.
    :type model_name: str
    :return: Initialized model instance.
    :rtype: DeepSeekModel | CodeGemmaModel | QwenModel | CodeLlamaModel
    """
    model = ModelFactory.create_model(
        model_type=model_type,
        model_name=model_name
    )
    model.load_model()

    return model
