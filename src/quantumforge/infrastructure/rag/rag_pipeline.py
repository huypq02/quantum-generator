from src.quantumforge.domain import (
    IRetriever,
    RetrieverConfig
)


class RAGPipeline:
    def __init__(self, retriever: IRetriever):
        self.retriever = retriever

    def get_context(self, query: str, config: RetrieverConfig) -> str:
        """
        Retrieve relevant context.
        
        :param query: User input.
        :param config: Retriever configuration.
        :return: Relevant context.
        """
        docs = self.retriever.retrieve(query, config)
        context_text = "\n\n".join(d.page_content for d in docs)

        return context_text
