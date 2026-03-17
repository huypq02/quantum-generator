from BCEmbedding.tools.langchain import BCERerank
from quantumgenerator.domain.interfaces import IReranker


class BCEReranker(IReranker):
    """Factory-style adapter for creating a BCERerank instance."""

    def __init__(self, model_name: str):
        """
        Initialize the reranker adapter.
        
        :param model_name: Pretrained BCEmbedding reranker model name or local path.
        :type model_name: str
        """
        self.model_name = model_name
    
    def rank(self, kwargs):
        """
        Create and return a configured ``BCERerank`` instance.
        
        This method does not execute reranking directly; it instantiates
        the underlying reranker object configured with ``self.model_name``.

        :return: Initialized BCERerank object.
        :rtype: BCERerank
        :raises RuntimeError: If BCERerank initialization fails.
        """
        try:
            return BCERerank(model=self.model_name, **kwargs)

        except Exception as e:
            print(f"Unexpected error while reranking model: {e}")
            raise RuntimeError("Unexpected error while reranking model")
