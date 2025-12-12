import torch
from transformers import AutoModelForCausalLM, GemmaTokenizer, BitsAndBytesConfig
from .base import BaseModel

class CodeGemmaModel(BaseModel):
    def load_model(self) -> None:
        """Load CodeGemma model."""
        quantize = self.config.get("quantize", False)

        self.tokenizer = GemmaTokenizer.from_pretrained(
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
                trust_remote_code=True
            )
            
        self.model.eval()

    def generate(
            self, 
            prompt: str, 
            max_new_tokens: int = 512, 
            temperature: float = 0.7, 
            top_p: float = 0.9,
            **kwargs
    ) -> str:
        """Generate text from a prompt."""

        input_text = prompt
        input_ids = self.tokenizer(input_text, return_tensors="pt")

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

        return self.tokenizer.decode(outputs[0])