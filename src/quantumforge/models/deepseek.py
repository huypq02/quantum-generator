import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class DeepSeekModel(BaseModel):
    def load_model(self) -> None:
        quantize = self.config.get("quantize", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mode_name, 
            trust_remote_code=True
        )

        if quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mode_name, 
                load_in_4bit=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mode_name, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).cuda()
