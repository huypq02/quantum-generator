from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import os
from .model_loader import load_model
from .data_loader import load_data
from quantumgenerator.domain import ITrainer, TrainingSession, TrainingResult


class LoRATrainer(ITrainer):
    """LoRA (Low-Rank Adaptation) trainer for fine-tuning models."""

    def train(self, session: TrainingSession) -> TrainingResult:
        """
        Execute the LoRA fine-tuning training pipeline.
        
        :param session: Training session configuration.
        :type session: TrainingSession
        :return: Training result containing adapter path and model name.
        :rtype: TrainingResult
        :raises RuntimeError: If fine-tuning fails.
        """
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

            return TrainingResult(
                adapter_path=model_lora,
                model_name=session.model_name
            )

        except Exception as e:
            print(f"Unexpected error while fine-tuning {e}")
            raise RuntimeError("Unexpected error while fine-tuning")
        
    def load_model(self, session: TrainingSession, result: TrainingResult):
        """
        Load a trained model with LoRA adapter.
        
        :param session: Training session configuration.
        :type session: TrainingSession
        :param result: Training result containing adapter information.
        :type result: TrainingResult
        :return: Loaded model with adapter.
        """
        loader = load_model(
            session.parameter.get("model_type"),
            session.model_name
        )
        loader.model = PeftModel.from_pretrained(loader.model, result.adapter_path)
        return loader
