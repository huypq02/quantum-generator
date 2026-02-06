from transformers import TrainingArguments, pipeline
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import os
from .model_loader import load_model
from .data_loader import load_data
from src.quantumforge.domain import ITrainer


class LoRATrainer(ITrainer):
    def __init__(
            self,
            config: dict
    ):
        self.config = config
    
    def train(self, dataset, config):
        model = load_model(config.get("model_type"), config.get("model_name"))

        model_lora = os.path.join("models", model.model_name)
        if not os.path.exists(model_lora):
            # Train the model if adapter doesn't exist
            training_argumnents = TrainingArguments(
                output_dir=config.get("output_dir"),
                per_device_train_batch_size=config.get("per_device_train_batch_size"),
                max_steps=config.get("max_steps")
            )
            sft_trainer = SFTTrainer(
                model=model.model,
                args=training_argumnents,
                train_dataset=dataset,
                peft_config=LoraConfig(
                    task_type=config.get("lora_task_type"), 
                    r=config.get("lora_r"), 
                    lora_alpha=config.get("lora_alpha"), 
                    lora_dropout=config.get("lora_dropout")
                ),
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

        return model


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
    trainer.train(
        dataset=load_data(),
        config=trainer.config
    )
