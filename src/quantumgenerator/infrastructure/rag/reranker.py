from sentence_transformers import CrossEncoder
from quantumgenerator.domain.interfaces import IReranker


class CrossEncoderReranker(IReranker):
    """Reranker implementation using a SentenceTransformers CrossEncoder model."""

    def __init__(self, model_name: str):
        """
        Initialize the cross-encoder reranker.
        
        :param model_name: Pretrained CrossEncoder model name or local path.
        :type model_name: str
        """
        self.reranker = CrossEncoder(model_name)
    
    def rank(self, query: str, docs: list[str]) -> list[str]:
        """
        Apply reranker to rank and sort documents by relevance.
        
        :param query: User input query.
        :type query: str
        :param docs: Retrieved documents to rerank.
        :type docs: list[str]
        :return: Documents sorted by relevance score in descending order.
        :rtype: list[str]
        """
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs)

        results = list(zip(scores, docs))
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results]
