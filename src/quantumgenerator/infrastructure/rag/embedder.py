from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = {
    "minilm-l6":"sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base":"sentence-transformers/all-mpnet-base-v2", # 768 dimensions
    "bge-small":"BAAI/bge-small-en-v1.5",                   # Better for RAG
    "para-multi-minilm":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

class EmbeddingModel:
    """Embedding model wrapper for document vectorization."""

    def __init__(self, model_type: str):
        """
        Initialize the embedding model.
        
        :param model_type: Type of embedding model to use.
        :type model_type: str
        """
        self.model_type = model_type
    
    def embed(self) -> HuggingFaceEmbeddings:
        """
        Choose and initialize the embedding model.
        
        :return: Initialized embedding model.
        :rtype: HuggingFaceEmbeddings
        :raises RuntimeError: If embedding model initialization fails.
        """
        try:
            return HuggingFaceEmbeddings(model_name=MODEL_NAME[self.model_type])
        
        except Exception as e:
            print(f"Unexpected error while embedding model: {e}")
            raise RuntimeError("Unexpected error while embedding model")
