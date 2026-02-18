from dataclasses import dataclass
from src.quantumforge.domain.entities.retriever_config import RetrieverConfig


@dataclass
class GenerateQuantumCodeRequest:
    """Request DTO for quantum code generation."""
    query: str
    retriever_config: RetrieverConfig = None
