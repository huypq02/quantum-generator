from quantumgenerator.domain import RetrieverConfig, RAGPipeline
from .factory import RetrieverFactory


class RAGPipelineImpl(RAGPipeline):
    """RAG (Retrieval-Augmented Generation) pipeline for context retrieval."""
    
    def get_context(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context for a given query.
        
        :param query: User input query.
        :type query: str
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Retrieved context as a single string.
        :rtype: str
        """
        retriever = RetrieverFactory().create_retriever(
            config.retriever_type, 
            embedding_model=config.embedder
        )
        docs = retriever.retrieve(query, config)
        context_text = "\n\n".join(d.page_content for d in docs)

        return context_text
