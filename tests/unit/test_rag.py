import unittest
from src.quantumforge.rag.retriever import load_retriever


class TestRag(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def test_load_retriever(self):
        """
        Docstring for loading retriever for RAG pipeline.
        """
        retriever = load_retriever()
        self.assertIsNotNone(retriever, "Retriever should be loaded.")
