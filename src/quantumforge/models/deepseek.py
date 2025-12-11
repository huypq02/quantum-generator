import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import BaseModel

class DeepSeekModel(BaseModel):
    def load_model(self) -> None:
        """Load DeepSeek model."""
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
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=dtype,
                device_map="auto"
            )
            
        self.model.eval()

    def tokenize(self, text: str) -> list:
        """Tokenize input text."""
        return self.tokenizer.encode(text)
    
    def generate(
            self, 
            prompt, 
            max_new_tokens = 512, 
            temperature = 0.7, 
            top_p = 0.9, 
            **kwargs
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer.apply_chat_template(
            messages=prompt, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
            top_k=50,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
