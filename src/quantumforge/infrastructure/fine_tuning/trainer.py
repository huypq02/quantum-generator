from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import os
from .model_loader import load_model
from .data_loader import load_data
from src.quantumforge.domain import ITrainer, TrainingSession


class LoRATrainer(ITrainer):
    def train(self, session: TrainingSession):
        try:
            loader = load_model(
                session.parameter.get("model_type"),
                session.model_name
            )
            train_dataset = load_data(session.data_id)
            model_lora = os.path.join("models", loader.model_name)
            if not os.path.exists(model_lora):
                # Train the model if adapter doesn't exist
                training_argumnents = TrainingArguments(
                    output_dir=session.output_path,
                    per_device_train_batch_size=session.parameter.get("per_device_train_batch_size"),
                    max_steps=session.parameter.get("max_steps")
                )
                sft_trainer = SFTTrainer(
                    model=loader.model,
                    args=training_argumnents,
                    train_dataset=train_dataset,
                    peft_config=LoraConfig(
                        task_type=session.parameter.get("lora_task_type"), 
                        r=session.parameter.get("lora_r"), 
                        lora_alpha=session.parameter.get("lora_alpha"), 
                        lora_dropout=session.parameter.get("lora_dropout")
                    ),
                )

                sft_trainer.train()

                # Save the LoRA adapter
                sft_trainer.model.save_pretrained(model_lora)
                print(f"Model saved to {model_lora}")
                
                # Update instance.model to use the trained model
                loader.model = sft_trainer.model
            else:
                # Load the existing LoRA adapter
                loader.model = PeftModel.from_pretrained(loader.model, model_lora)
                print(f"Model loaded from {model_lora}")

            print("New model", sft_trainer.model)
            print("Saved Model", PeftModel.from_pretrained(loader.model, model_lora))

        except Exception as e:
            print(f"Unexpected error while fine-tuning {e}")
            raise RuntimeError("Unexpected error while fine-tuning")
        
        return None
