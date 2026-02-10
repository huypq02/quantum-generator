import os
from src.quantumforge.infrastructure.rag.retriever import load_retriever


class TestRag():
    def test_load_retriever(self):
        """
        Docstring for loading retriever for RAG pipeline.
        """
        retriever = load_retriever()
        assert retriever is not None, "Retriever should be loaded."
