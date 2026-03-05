from dataclasses import dataclass
from typing import Optional
from quantumgenerator.domain.entities.retriever_config import RetrieverConfig


@dataclass
class GenerateQuantumCodeRequest:
    """
    Request DTO for quantum code generation.
    
    :param query: User query for code generation.
    :type query: str
    :param retriever_config: Optional configuration for the retriever.
    :type retriever_config: Optional[RetrieverConfig]
    """
    query: str
    retriever_config: Optional[RetrieverConfig] = None
