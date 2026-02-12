from langchain_chroma import Chroma
import os
from .embedder import EmbeddingModel
from src.quantumforge.domain import (
    IRetriever,
    RetrieverConfig
)

class ChromaRetriever(IRetriever):
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedder = embedding_model

    def retrieve(self, query: str, config: RetrieverConfig):
        """
        Find closest documents to embedded user query.
        
        :param config: Retriever configuration.
        :return: The most relevant document to the query.
        """
        try:
            # Using HuggingFace embeddings
            embeddings = self.embedder.embed()

            # Check if a directory exists or not then create new
            os.makedirs(config.vectordb_path, exist_ok=True)

            # Create a Chroma vector database that keeps all vector in the local directory
            vectorstore = Chroma.from_documents(
                documents=config.documents,
                embedding=embeddings,
                persist_directory=config.vectordb_path
            )
            
            # Finding the most relevant document to the query
            retriever = vectorstore.as_retriever(
                search_type=config.search_type,
                search_kwargs=config.search_kwargs
            )
            
            return retriever.invoke(query)

        except Exception as e:
            print(f"Error while processing retriever: {e}")
            raise RuntimeError("Error while processing retriever")
