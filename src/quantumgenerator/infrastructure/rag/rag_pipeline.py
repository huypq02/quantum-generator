from quantumgenerator.domain import RetrieverConfig
from .factory import RetrieverFactory


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline for context retrieval."""

    def __init__(self, retriever_factory: RetrieverFactory):
        """
        Initialize the RAG pipeline.
        
        :param retriever_factory: Factory for creating retriever instances.
        :type retriever_factory: RetrieverFactory
        """
        self.retriever_factory = retriever_factory

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
        retriever = self.retriever_factory.create_retriever(
            config.retriever_type, 
            embedding_model=config.embedder
        )
        docs = retriever.retrieve(query, config)
        context_text = "\n\n".join(d.page_content for d in docs)

        return context_text
