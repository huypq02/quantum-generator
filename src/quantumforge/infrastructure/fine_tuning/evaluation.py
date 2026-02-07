from src.quantumforge.infrastructure.generators import (
    DeepSeekModel,
    CodeGemmaModel,
    QwenModel,
    CodeLlamaModel
)


def evaluate(
        user_prompt: str, 
        model: DeepSeekModel | CodeGemmaModel | QwenModel | CodeLlamaModel
):
    try:
        text_generation = model.generate(user_prompt)

        return text_generation
    except Exception as e:
        print(f"An unexpected error occurred while evaluating the model: {e}")
        raise RuntimeError("An unexpected error occurred while evaluating the model.")
