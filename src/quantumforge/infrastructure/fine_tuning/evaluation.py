from .trainer import LoRATrainer
from .data_loader import load_data
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

        print(text_generation)
    except Exception as e:
        print(f"An unexpected error occurred while evaluating the model: {e}")
        raise RuntimeError("An unexpected error occurred while evaluating the model.")


if __name__ == "__main__":
    trainer = LoRATrainer(
        config={
            "model_type":"codegemma",
            "model_name":"google/codegemma-2b",
            "output_dir":"./checkpoints",
            "per_device_train_batch_size":4,
            "max_steps":100,
            "lora_task_type":"CAUSAL_LM",
            "lora_r":64, 
            "lora_alpha":16, 
            "lora_dropout":0.1
        }
    )
    model = trainer.train(
        dataset=load_data(),
        config=trainer.config
    )

    evaluate(
        user_prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm",
        model=model
    )
