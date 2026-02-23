from typing import Optional
from functools import lru_cache
from quantumgenerator.infrastructure import (
    ModelFactory,
    RetrieverFactory,
    SystemClock,
)
from quantumgenerator.application import (
    GenerateQuantumCodeUseCase,
    CodeGenerationService,
)


class DIContainer:
    """Centralize dependency injection container."""

    @lru_cache(maxsize=1)
    def get_model_factory(self):
        return ModelFactory()
    
    @lru_cache(maxsize=1)
    def get_retriever_factory(self):
        return RetrieverFactory()
    
    @lru_cache(maxsize=1)
    def get_system_clock(self):
        return SystemClock()

    def get_code_generation_service(
            self,
            model_type: str = "codegemma", 
            model_name: str = "google/codegemma-2b", 
            retriever_type: str = "chroma"
    ) -> CodeGenerationService:
        generator = self.get_model_factory().create_model(
            model_type, 
            model_name=model_name
        )
        retriever = self.get_retriever_factory().create_retriever(retriever_type)
        clock = self.get_system_clock()

        use_case = GenerateQuantumCodeUseCase(generator, retriever)
        return CodeGenerationService(use_case, clock)


@lru_cache(maxsize=1)
def create_container() -> DIContainer:
    return DIContainer()
