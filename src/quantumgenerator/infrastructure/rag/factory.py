from .chroma_retriever import ChromaRetriever


class RetrieverFactory:
    """Factory for creating retriever instances."""

    def create_retriever(self, retriever_type: str, **kwargs):
        """
        Create a specific retriever instance.
        
        :param retriever_type: Type of retriever to create (e.g., 'chroma').
        :type retriever_type: str
        :param kwargs: Additional configuration parameters for the retriever.
        :return: Retriever instance.
        :raises ValueError: If retriever type is unknown.
        """"
        if retriever_type == "chroma":
            return ChromaRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
