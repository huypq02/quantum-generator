import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel

class DeepSeekModel(BaseModel):
    def load_model(self) -> None:
        quantize = self.config.get("quantize", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )

        if quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                load_in_4bit=True,
                device_map="auto"
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=dtype,
                device_map="auto"
            )
            
        self.model.eval()