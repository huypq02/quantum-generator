from transformers import TrainingArguments, pipeline
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
import os
from src.quantumforge.models.factory import ModelFactory

model_name = "google/codegemma-2b"
model = ModelFactory.create_model(
    model_type="codegemma",
    model_name=model_name
)
model.load_model()

model_lora = os.path.join("models", model.model_name)
if not os.path.exists(model_lora):
    # Train the model if adapter doesn't exist
    training_argumnents = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        max_steps=100
    )
    sft_trainer = SFTTrainer(
        model=model.model,
        args=training_argumnents,
        train_dataset=load_dataset("webxos/OPENCLAW_quantum_dataset", split="train"),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
    )

    sft_trainer.train()

    # Save the LoRA adapter
    sft_trainer.model.save_pretrained(model_lora)
    print(f"Model saved to {model_lora}")
    
    # Update instance.model to use the trained model
    model.model = sft_trainer.model
else:
    # Load the existing LoRA adapter
    model.model = PeftModel.from_pretrained(model.model, model_lora)
    print(f"Model loaded from {model_lora}")

user_prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
text_generation = model.generate(user_prompt)

print(text_generation)
