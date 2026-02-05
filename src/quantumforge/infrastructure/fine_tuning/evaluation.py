from .trainer import train
from src.quantumforge.models.deepseek import DeepSeekModel
from src.quantumforge.models.codegemma import CodeGemmaModel
from src.quantumforge.models.qwen import QwenModel
from src.quantumforge.models.codellama import CodeLlamaModel

def evaluate(
        user_prompt: str, 
        model: DeepSeekModel | CodeGemmaModel | QwenModel | CodeLlamaModel
):
    
    text_generation = model.generate(user_prompt)

    print(text_generation)


if __name__ == "__main__":
    model = train(
        model_type="codegemma",
        model_name="google/codegemma-2b"
    )

    evaluate(
        user_prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm",
        model=model
    )