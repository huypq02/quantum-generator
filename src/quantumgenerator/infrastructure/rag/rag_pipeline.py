from langchain_classic.retrievers import ContextualCompressionRetriever
from quantumgenerator.domain import RetrieverConfig, RAGPipeline
from .factory import RetrieverFactory
from .reranker import CrossEncoderReranker


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

        reranker = CrossEncoderReranker(config).rank()

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )
        docs = compression_retriever.invoke(query)
        context_text = "\n\n".join(d.page_content for d in docs)
        return context_text
