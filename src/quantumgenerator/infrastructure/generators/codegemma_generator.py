import torch
from transformers import AutoModelForCausalLM, GemmaTokenizer, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from quantumgenerator.domain.interfaces.generator import IGenerator


class CodeGemmaModel(IGenerator):
    """CodeGemma model implementation for code generation."""

    def load_model(self) -> None:
        """
        Load CodeGemma model.
        
        :return: None
        :rtype: None
        :raises RuntimeError: If model loading fails.
        """
        try:
            # Load environment variables
            load_dotenv()
            
            hf_token = os.environ.get("HF_TOKEN")
            quantize = self.config.get("quantize", False)

            self.tokenizer = GemmaTokenizer.from_pretrained(
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
                    token=hf_token,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    token=hf_token,
                    trust_remote_code=True
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
        """
        Generate text from a prompt.
        
        :param prompt: Input prompt for text generation.
        :type prompt: str
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param top_p: Nucleus sampling parameter.
        :type top_p: float
        :param kwargs: Additional generation parameters.
        :return: Generated text.
        :rtype: str
        :raises RuntimeError: If text generation fails.
        """
        try:
            # Auto-load model if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            input_text = prompt
            input_ids = self.tokenizer(
                input_text, 
                return_tensors="pt"
            ).to(self.model.device)
            
            input_length = input_ids.input_ids.shape[-1]

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,
                **kwargs
            )

            # Decode only the newly generated tokens (exclude input)
            generated_tokens = outputs[0][input_length:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        except Exception as e:
            print(f"An unexpected error occurred while generating text from the model: {e}")
            raise RuntimeError("An unexpected error occurred while generating text from the model")
