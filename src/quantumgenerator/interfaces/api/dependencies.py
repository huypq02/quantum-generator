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
    """
    Centralized dependency injection container.
    """

    @lru_cache(maxsize=1)
    def get_model_factory(self):
        """
        Get or create the model factory instance.
        
        :return: Model factory instance.
        :rtype: ModelFactory
        """
        return ModelFactory()
    
    @lru_cache(maxsize=1)
    def get_retriever_factory(self):
        """
        Get or create the retriever factory instance.
        
        :return: Retriever factory instance.
        :rtype: RetrieverFactory
        """
        return RetrieverFactory()
    
    @lru_cache(maxsize=1)
    def get_system_clock(self):
        """
        Get or create the system clock instance.
        
        :return: System clock instance.
        :rtype: SystemClock
        """
        return SystemClock()

    def get_code_generation_service(
            self,
            model_type: str = "codegemma", 
            model_name: str = "google/codegemma-2b", 
            retriever_type: str = "chroma"
    ) -> CodeGenerationService:
        """
        Get code generation service with configured dependencies.
        
        :param model_type: Type of model to use for generation.
        :type model_type: str
        :param model_name: Name or path of the model.
        :type model_name: str
        :param retriever_type: Type of retriever to use.
        :type retriever_type: str
        :return: Configured code generation service.
        :rtype: CodeGenerationService
        """
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
    """
    Create and cache the dependency injection container.
    
    :return: DI container instance.
    :rtype: DIContainer
    """
    return DIContainer()
