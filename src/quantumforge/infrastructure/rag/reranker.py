from sentence_transformers import CrossEncoder
from src.quantumforge.domain.interfaces import IReranker


class CrossEncoderReranker(IReranker):
    def __init__(self, model_name):
        self.reranker = CrossEncoder(model_name)
    
    def rank(self, query: str, docs: list[str]) -> list[str]:
        """
        Apply Reranker and rank and display results.
        
        :param query: User input
        :type query: str
        :param docs: Retrieval documents
        :type docs: list[str]
        :return: List of reranked documents
        :rtype: list[str]
        """
        pairs = [(query, doc.page_content) for doc in self.docs]
        scores = self.reranker.predict(pairs)

        results = list(zip(scores, docs))
        results.sort(key=lambda x: x[0], reverse=True)
        return results
