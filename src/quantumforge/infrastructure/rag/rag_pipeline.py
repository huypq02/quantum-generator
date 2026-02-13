from src.quantumforge.domain import (
    IRetriever,
    RetrieverConfig
)
from .embedder import EmbeddingModel
from .chroma_retriever import ChromaRetriever


class RAGPipeline:
    def __init__(self, retriever: ChromaRetriever):
        self.retriever = retriever

    def get_context(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context.
        
        :param query: User input.
        :param config: Retriever configuration.
        :return: Relevant context.
        """
        initial_retriever = self.retriever(config.embedder)
        docs = initial_retriever.retrieve(query, config)
        context_text = "\n\n".join(d.page_content for d in docs)

        return context_text
