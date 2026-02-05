from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingModel:
    @staticmethod
    def create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        if model_name == "minilm-l6":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif model_name == "mpnet-base":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
            )
        elif model_name == "bge-small":
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"  # Better for RAG
            )
        elif model_name == "para-multi-minilm":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        else:
            raise ValueError(f"Unknown embedding model name: {model_name}")
