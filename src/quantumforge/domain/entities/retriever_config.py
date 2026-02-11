from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    vectordb_path: str
    documents: str
    search_type: str = "mmr" # Using Maximal Marginal Relevance algorithm
    search_kwargs: dict = {
        'k':1,               # Top 1 most relevant document
        'lambda_mult':0.7,   # 70% focus on relevance, 30% on diversity
    }
