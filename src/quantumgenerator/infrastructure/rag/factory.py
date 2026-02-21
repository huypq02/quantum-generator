from .chroma_retriever import ChromaRetriever


class RetrieverFactory:
    def create_retriever(self, retriever_type: str, **kwargs):
        """Create a specific retriever."""
        if retriever_type == "chroma":
            return ChromaRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
