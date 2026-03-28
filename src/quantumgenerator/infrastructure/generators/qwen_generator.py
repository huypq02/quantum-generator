import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from quantumgenerator.domain.interfaces.generator import IGenerator


class QwenModel(IGenerator):
    """Qwen model implementation for code generation."""

    def load_model(self) -> None:
        """
        Load Qwen model.
        
        :return: None
        :rtype: None
        :raises RuntimeError: If model loading fails.
        """
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
                    torch_dtype=dtype,
                    device_map="auto"
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
            system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
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
        :param system_prompt: System prompt for the model.
        :type system_prompt: str
        :param kwargs: Additional generation parameters.
        :return: Generated text.
        :rtype: str
        :raises RuntimeError: If text generation fails.
        """
        try:
            # Auto-load model if not already loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
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
                do_sample=True,
                repetition_penalty=1.2,
                **kwargs
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        except Exception as e:
            print(f"An unexpected error occurred while generating text from the model: {e}")
            raise RuntimeError("An unexpected error occurred while generating text from the model")
