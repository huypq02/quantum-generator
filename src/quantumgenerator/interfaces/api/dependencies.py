from typing import Optional
from functools import lru_cache
from quantumgenerator.domain import RetrieverConfig
from quantumgenerator.infrastructure.rag import (
    RAGPipelineImpl,
    EmbeddingModel,
    chunking,
    load_data,
)
from quantumgenerator.infrastructure.generators import ModelFactory
from quantumgenerator.infrastructure.time import SystemClock
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
    def get_retriever_config(self):
        """
        Get retriever configuration.
        
        :return: Retriever configuration.
        :rtype: RetrieverConfig
        """
        # TODO: should apply loading configuration from yaml file
        return RetrieverConfig(
            retriever_type="chroma",
            vectordb_path="data/vectordb/chroma",
            documents=chunking(
                encoding_name="cl100k_base",
                chunk_size=200,
                chunk_overlap=40,
                doc_list=load_data("data/quantum_docs/general/Intro-to-AI-notes.pdf")
            ),  # from request or service
            embedder=EmbeddingModel("minilm-l6"),   # from container
            vectordb_mode="local",
            collection_name="quantum_docs",
            search_type="mmr",
            search_kwargs={"k": 1, "lambda_mult": 0.7}
        )
    
    @lru_cache(maxsize=1)
    def get_rag_pipeline_impl(self):
        """
        Get or create the RAG Pipeline instance.
        
        :return: RAG Pipeline instance.
        :rtype: RAGPipelineImpl
        """
        return RAGPipelineImpl()
    
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
        retriever_config = self.get_retriever_config()
        rag_pipeline = self.get_rag_pipeline_impl()
        clock = self.get_system_clock()

        use_case = GenerateQuantumCodeUseCase(
            generator, retriever_config, rag_pipeline
        )
        return CodeGenerationService(use_case, clock)


@lru_cache(maxsize=1)
def create_container() -> DIContainer:
    """
    Create and cache the dependency injection container.
    
    :return: DI container instance.
    :rtype: DIContainer
    """
    return DIContainer()
