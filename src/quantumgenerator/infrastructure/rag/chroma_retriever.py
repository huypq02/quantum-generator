from langchain_chroma import Chroma
import os
import chromadb
from .embedder import EmbeddingModel
from quantumgenerator.domain import (
    IRetriever,
    RetrieverConfig
)

class ChromaRetriever(IRetriever):
    """Chroma-based document retriever implementation."""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize the Chroma retriever.
        
        :param embedding_model: Embedding model for vectorization.
        :type embedding_model: EmbeddingModel
        """
        self.embedder = embedding_model

    def _create_client(self, config: RetrieverConfig):
        """Create a remote Chroma client when configured for remote mode."""
        if config.host is None or config.port is None:
            raise ValueError("Remote Chroma requires both host and port")

        return chromadb.HttpClient(
            host=config.host,
            port=config.port,
            ssl=config.ssl,
        )

    def index_documents(self, config: RetrieverConfig) -> None:
        """
        Build or refresh a persisted Chroma index from source documents.

        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :raises RuntimeError: If indexing process fails.
        """
        try:
            embeddings = self.embedder.embed()
            if config.vectordb_mode == "local":
                os.makedirs(config.vectordb_path, exist_ok=True)

                Chroma.from_documents(
                    documents=config.documents,
                    embedding=embeddings,
                    persist_directory=config.vectordb_path,
                    collection_name=config.collection_name,
                )
            else:
                client = self._create_client(config)
                Chroma.from_documents(
                    documents=config.documents,
                    embedding=embeddings,
                    client=client,
                    collection_name=config.collection_name,
                )
        except Exception as e:
            print(f"Error while indexing documents: {e}")
            raise RuntimeError("Error while indexing documents")

    def retrieve_context(self, config: RetrieverConfig):
        """
        Find closest documents to embedded user query.
        
        :param config: Retriever configuration settings.
        :type config: RetrieverConfig
        :return: The most relevant documents to the query.
        :raises RuntimeError: If retrieval process fails.
        """
        try:
            embeddings = self.embedder.embed()

            if config.vectordb_mode == "local":
                # Query against an existing persisted index.
                vectorstore = Chroma(
                    persist_directory=config.vectordb_path,
                    embedding_function=embeddings,
                    collection_name=config.collection_name,
                )
            else:
                client = self._create_client(config)
                vectorstore = Chroma(
                    client=client,
                    embedding_function=embeddings,
                    collection_name=config.collection_name,
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
