import os
import pytest
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig
from quantumgenerator.infrastructure.rag import (
    load_data,
    chunking,
    EmbeddingModel,
    RAGPipelineImpl,
)


class TestRag():
    @pytest.fixture(autouse=True)
    def setup(self):
        root_dir = os.getcwd()
        self.docs_dir = os.path.join(root_dir, "data", "quantum_docs")
        self.vectordb_dir = os.path.join(root_dir, "data", "vectordb", "faiss")
        self.default_file_path = os.path.join("general", "2024-wttc-introduction-to-ai.pdf")
        self.embedding_type = "minilm-l6"
        self.search_type = "mmr"
        self.search_kwargs = { 'k': 10,'lambda_mult': 0.7 }
        self.user_input = "What is unsupervised learning?"
        self.retriever_type = "faiss"

        self.loader = load_data(os.path.join(self.docs_dir, self.default_file_path))
        self.chunker = chunking(
            encoding_name="cl100k_base",
            chunk_size=300,
            chunk_overlap=60,
            doc_list=self.loader
        )
        self.embedder = EmbeddingModel(self.embedding_type)
    
    def test_index_documents(self):
        """
        Docstring for indexing documents for RAG pipeline.
        """
        os.makedirs(self.vectordb_dir, exist_ok=True)
        config = RetrieverConfig(
            retriever_type=self.retriever_type,
            vectordb_path=self.vectordb_dir,
            documents=self.chunker,
            embedder=self.embedder,
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )
        pipeline = RAGPipelineImpl()
        pipeline.index_documents(config)

    def test_compression_retriever(self):
        """
        Docstring for loading retriever for RAG pipeline.
        """
        os.makedirs(self.vectordb_dir, exist_ok=True)
        config = RetrieverConfig(
            retriever_type=self.retriever_type,
            vectordb_path=self.vectordb_dir,
            documents=self.chunker,
            embedder=self.embedder,
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )
        pipeline = RAGPipelineImpl()
        # pipeline.index_documents(config)
        context = pipeline.compression_retriever(
            query=self.user_input,
            config=config,
        )

        assert len(context) > 0, "Result should be not empty."
