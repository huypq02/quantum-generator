import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from quantumgenerator.domain.interfaces.generator import IGenerator


class CodeLlamaModel(IGenerator):
    def load_model(self) -> None:
        """Load CodeLlama model."""
        try:
            # Load environment variables
            load_dotenv()
            
            hf_token = os.environ.get("HF_TOKEN")
            quantize = self.config.get("quantize", False)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True

            )

            if quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    device_map="auto",
                    token=hf_token,
                    quantization_config=quantization_config
                )
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    dtype=dtype,
                    device_map="auto",
                    token=hf_token
                )
                
            self.model.eval()
        
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            raise RuntimeError("An unexpected error occurred while loading the model")

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
            input_text = prompt
            input_ids = self.tokenizer(
                input_text, 
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

            return self.tokenizer.decode(outputs[0])
        
        except Exception as e:
            print(f"An unexpected error occurred while generating text from the model: {e}")
            raise RuntimeError("An unexpected error occurred while generating text from the model")
