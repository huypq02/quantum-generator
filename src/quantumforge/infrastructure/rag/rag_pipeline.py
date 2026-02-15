from src.quantumforge.domain import RetrieverConfig
from .factory import RetrieverFactory


class RAGPipeline:
    def __init__(self, retriever_factory: RetrieverFactory):
        self.retriever_factory = retriever_factory

    def get_context(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context.
        
        :param query: User input.
        :param config: Retriever configuration.
        :return: Relevant context.
        """
        retriever = self.retriever_factory.create_retriever(
            config.retriever_type, 
            embedding_model=config.embedder
        )
        docs = retriever.retrieve(query, config)
        context_text = "\n\n".join(d.page_content for d in docs)

        return context_text
