from langchain_community.vectorstores import FAISS
import os
from .embedder import EmbeddingModel
from quantumgenerator.domain import (
    IRetriever,
    RetrieverConfig
)

class FAISSRetriever(IRetriever):
    """FAISS-based document retriever implementation."""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize the FAISS retriever.
        
        :param embedding_model: Embedding model for vectorization.
        :type embedding_model: EmbeddingModel
        """
        self.embedder = embedding_model

    def index_documents(self, config: RetrieverConfig) -> None:
        """
        Build a FAISS index from source documents.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :raises RuntimeError: If indexing process fails.
        """
        try:
            embeddings = self.embedder.embed()
            os.makedirs(config.vectordb_path, exist_ok=True)

            FAISS.from_documents(
                documents=config.documents,
                embedding=embeddings,
                persist_directory=config.vectordb_path,
            )
        except Exception as e:
            print(f"Error while indexing documents: {e}")
            raise RuntimeError("Error while indexing documents")

    def retrieve_context(self, config: RetrieverConfig):
        """
        Retrieve the most relevant documents for a user query.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: Documents most relevant to the query.
        :rtype: list
        :raises RuntimeError: If retrieval process fails.
        """
        try:
            embeddings = self.embedder.embed()

            # Query against an existing persisted index.
            vectorstore = FAISS(
                persist_directory=config.vectordb_path,
                embedding_function=embeddings,
            )
            
            # Finding the most relevant document to the query
            retriever = vectorstore.as_retriever(
                search_type=config.search_type,
                search_kwargs=config.search_kwargs
            )
            return retriever

        except Exception as e:
            print(f"Error while retrieving context: {e}")
            raise RuntimeError("Error while retrieving context")
