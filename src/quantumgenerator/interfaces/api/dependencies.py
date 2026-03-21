from pathlib import Path
from functools import lru_cache
from quantumgenerator.domain import RetrieverConfig
from quantumgenerator.infrastructure.config.config import load_config
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


CONFIG_PATH = Path.cwd() / "config" / "config.yaml"


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
        config = load_config(str(CONFIG_PATH)) or {}
        retriever_cfg = config.get("retriever", {})

        documents_cfg = retriever_cfg.get("documents", {})
        chunking_cfg = retriever_cfg.get("chunking", {})

        document_paths = documents_cfg.get(
            "paths",
            ["data/quantum_docs/general/Intro-to-AI-notes.pdf"],
        )
        if isinstance(document_paths, str):
            document_paths = [document_paths]

        loaded_documents = []
        for document_path in document_paths:
            loaded_documents.extend(load_data(document_path))

        search_kwargs = retriever_cfg.get(
            "search_kwargs",
            {"k": 10, "lambda_mult": 0.7},
        )

        return RetrieverConfig(
            retriever_type=retriever_cfg.get("retriever_type", "chroma"),
            vectordb_path=retriever_cfg.get("vectordb_path", "data/vectordb/chroma"),
            documents=chunking(
                encoding_name=chunking_cfg.get("encoding_name", "cl100k_base"),
                chunk_size=chunking_cfg.get("chunk_size", 200),
                chunk_overlap=chunking_cfg.get("chunk_overlap", 40),
                doc_list=loaded_documents,
            ),
            embedder=EmbeddingModel(retriever_cfg.get("embedder", "minilm-l6")),
            search_type=retriever_cfg.get("search_type", "mmr"),
            vectordb_mode=retriever_cfg.get("vectordb_mode", "local"),
            collection_name=retriever_cfg.get("collection_name", "quantum_docs"),
            host=retriever_cfg.get("host"),
            port=retriever_cfg.get("port"),
            ssl=retriever_cfg.get("ssl", False),
            search_kwargs=search_kwargs,
            rerank_model=retriever_cfg.get("rerank_model", "BAAI/bge-reranker-base"),
            rerank_top_n=retriever_cfg.get("rerank_top_n", 3),
            rerank_device=retriever_cfg.get("rerank_device"),
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
