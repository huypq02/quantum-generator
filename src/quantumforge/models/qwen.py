import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import BaseModel

class QwenModel(BaseModel):
    def load_model(self) -> None:
        """Load Qwen model."""
        quantize = self.config.get("quantize", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )

        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map="auto",
                quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype="auto",
                device_map="auto"
            )
            
        self.model.eval()

    def generate(
            self, 
            prompt: str, 
            max_new_tokens: int = 512, 
            temperature: float = 0.7, 
            top_p: float = 0.9,
            system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            **kwargs
    ) -> str:
        """Generate text from a prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
