import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker as LCCrossEncoderReranker
from quantumgenerator.domain.interfaces import IReranker
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


class CrossEncoderReranker(IReranker):
    """Factory-style adapter for creating a cross-encoder reranker instance."""

    def __init__(self, config: RetrieverConfig):
        """
        Initialize the reranker adapter.

        :param config: Retriever configuration containing reranker settings.
        :type config: RetrieverConfig
        """
        self.config = config

    def rank(self):
        """
        Create and return a configured LangChain cross-encoder reranker instance.

        :return: Initialized cross-encoder reranker object.
        :rtype: LCCrossEncoderReranker
        :raises RuntimeError: If reranker initialization fails.
        """
        try:
            device = self.config.rerank_device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

            model = HuggingFaceCrossEncoder(
                model_name=self.config.rerank_model,
                model_kwargs={"device": device},
            )

            return LCCrossEncoderReranker(model=model, top_n=self.config.rerank_top_n)

        except Exception as e:
            print(f"Unexpected error while reranking model: {e}")
            raise RuntimeError("Unexpected error while reranking model")
