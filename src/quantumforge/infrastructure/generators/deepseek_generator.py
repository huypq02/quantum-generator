import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base_generator import BaseModel

class DeepSeekModel(BaseModel):
    def load_model(self) -> None:
        """Load DeepSeek model."""
        try:
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
                    dtype=dtype,
                    device_map="auto"
                )
                
            self.model.eval()
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            raise
    
    def generate(
            self, 
            prompt: str, 
            max_new_tokens: int = 512, 
            temperature: float = 0.7, 
            top_p: float = 0.9,
            **kwargs
    ) -> str:
        """Generate text from a prompt."""
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
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
        
        except Exception as e:
            print(f"An unexpected error occurred while generating text from the model: {e}")
            raise
