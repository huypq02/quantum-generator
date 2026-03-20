from langchain_classic.retrievers import ContextualCompressionRetriever
from quantumgenerator.domain import RetrieverConfig, RAGPipeline
from .factory import RetrieverFactory
from .reranker import CrossEncoderReranker
from quantumgenerator.infrastructure.logger import setup_logging
import logging

logger = setup_logging(__name__)


class RAGPipelineImpl(RAGPipeline):
    """RAG (Retrieval-Augmented Generation) pipeline for context retrieval."""

    def index_documents(self, config: RetrieverConfig) -> None:
        """
        Build or refresh the configured retrieval index.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        """
        retriever = RetrieverFactory().create_retriever(
            config.retriever_type,
            embedding_model=config.embedder,
        )
        retriever.index_documents(config)
    
    def compression_retriever(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context for a given query.
        
        :param query: User input query.
        :type query: str
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Retrieved context as a single string.
        :rtype: str
        """
        retriever_type = RetrieverFactory().create_retriever(
            config.retriever_type, 
            embedding_model=config.embedder
        )
        retriever = retriever_type.retrieve_context(config)

        raw_docs = retriever.invoke(query)
        logger.debug("--- BEFORE RERANKER ---")
        for i, d in enumerate(raw_docs):
            logger.debug("[%d] %s", i, d.page_content[:200])
        logger.debug("--- END BEFORE RERANKER ---")

        reranker = CrossEncoderReranker(config).rank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )
        reranked_docs = compression_retriever.invoke(query)
        logger.debug("--- AFTER RERANKER ---")
        for i, d in enumerate(reranked_docs):
            logger.debug("[%d] %s", i, d.page_content[:200])
        logger.debug("--- END AFTER RERANKER ---")

        context_text = "\n\n".join(d.page_content for d in reranked_docs)
        return context_text
