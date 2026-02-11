from .embedder import EmbeddingModel
from src.quantumforge.domain import (
    IRetriever,
    RetrieverConfig
)
from langchain_chroma import Chroma


class ChromaRetriever(IRetriever):
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedder = embedding_model

    def retrieve(self, session: RetrieverConfig):
        try:
            # Using HuggingFace embeddings
            embeddings = self.embedder.embed()

            # Check if a directory exists or not then create new
            os.makedirs(session.vectordb_path, exist_ok=True)

            # Create a Chroma vector database that keeps all vector in the local directory
            vectorstore = Chroma.from_documents(documents=session.documents,
                                                embedding=embeddings,
                                                persist_directory=session.vectordb_path)
            
            # Finding the most relevant document to the query
            retriever = vectorstore.as_retriever(search_type=session.search_type,
                                                search_kwargs=session.search_kwargs)
            
            return retriever

        except Exception as e:
            print(f"Error while processing retriever: {e}")
            raise RuntimeError("Error while processing retriever")
