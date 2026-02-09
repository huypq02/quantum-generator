import pytest
from src.quantumforge.infrastructure.generators import (
    DeepSeekModel,
    CodeGemmaModel,
    QwenModel,
    CodeLlamaModel
)


class TestDeepSeekModel():
    @pytest.fixture(autouse=True)
    def set_up(self):
        """
        Docstring for setting up some essential parameters.
        """
        self.model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        self.model_type = DeepSeekModel(self.model_name)
        self.prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
        return self
    
    def test_load_model(self):
        """
        Docstring for the unit testing of the load DeepSeek model.
        """
        self.model_type.load_model()
        assert self.model_type.model is not None, "Model should be loaded"
        assert self.model_type.tokenizer is not None, "Tokenizer should be loaded"

    def test_generate(self):
        """
        Docstring for the unit testing of generating the DeepSeek model text.
        """
        self.model_type.load_model()
        text = self.model_type.generate(self.prompt, max_new_tokens=100)

        assert isinstance(text, str), "Generated text should be a string"
        assert len(text) > 0, "Generated text should not be empty"

class TestCodeGemmaModel():
    @pytest.fixture(autouse=True)
    def set_up(self):
        """
        Docstring for setting up some essential parameters.
        """
        self.model_name = "google/codegemma-2b"
        self.model_type = CodeGemmaModel(self.model_name)
        self.prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
    
    def test_load_model(self):
        """
        Docstring for the unit testing of the load CodeGemma model.
        """
        self.model_type.load_model()
        assert self.model_type.model is not None, "Model should be loaded"
        assert self.model_type.tokenizer is not None, "Tokenizer should be loaded"

    def test_generate(self):
        """
        Docstring for the unit testing of generating the CodeGemma model text.
        """
        self.model_type.load_model()
        text = self.model_type.generate(self.prompt, max_new_tokens=100)

        assert isinstance(text, str), "Generated text should be a string"
        assert len(text) > 0, "Generated text should not be empty"

class TestQwenModel():
    @pytest.fixture(autouse=True)
    def set_up(self):
        """
        Docstring for setting up some essential parameters.
        """
        self.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        self.model_type = QwenModel(self.model_name)
        self.prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
    
    def test_load_model(self):
        """
        Docstring for the unit testing of the load Qwen model.
        """
        self.model_type.load_model()
        assert self.model_type.model is not None, "Model should be loaded"
        assert self.model_type.tokenizer is not None, "Tokenizer should be loaded"

    def test_generate(self):
        """
        Docstring for the unit testing of generating the Qwen model text.
        """
        self.model_type.load_model()
        text = self.model_type.generate(self.prompt, max_new_tokens=100)

        assert isinstance(text, str), "Generated text should be a string"
        assert len(text) > 0, "Generated text should not be empty"

class TestCodeLlamaModel():
    @pytest.fixture(autouse=True)
    def set_up(self):
        """
        Docstring for setting up some essential parameters.
        """
        self.model_name = "meta-llama/CodeLlama-7b-hf"
        self.model_type = CodeLlamaModel(self.model_name)
        self.prompt = "Generate OpenQASM 3.0 code implementing Grover's algorithm"
    
    def test_load_model(self):
        """
        Docstring for the unit testing of the load CodeLlama model.
        """
        self.model_type.load_model()
        assert self.model_type.model is not None, "Model should be loaded"
        assert self.model_type.tokenizer is not None, "Tokenizer should be loaded"

    def test_generate(self):
        """
        Docstring for the unit testing of generating the CodeLlama model text.
        """
        self.model_type.load_model()
        text = self.model_type.generate(self.prompt, max_new_tokens=100)

        assert isinstance(text, str), "Generated text should be a string"
        assert len(text) > 0, "Generated text should not be empty"
