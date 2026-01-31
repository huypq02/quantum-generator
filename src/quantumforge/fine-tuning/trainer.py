from transformers import TrainingArguments, pipeline
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
from src.quantumforge.models.factory import ModelFactory

training_argumnents = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    max_steps=100
)

model_name = "google/codegemma-2b"
instance = ModelFactory.create_model(
    model_type="codegemma",
    model_name=model_name
)

instance.load_model()
sft_trainer = SFTTrainer(
    model=instance.model,
    args=training_argumnents,
    train_dataset=load_dataset("webxos/OPENCLAW_quantum_dataset", split="train"),
    peft_config=LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=16, lora_dropout=0.1),
)

sft_trainer.train()


user_prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
text_generation = instance.generate(user_prompt)

print(text_generation)