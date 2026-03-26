"""Domain Layer - Core business logic, entities, and interfaces."""
from .entities import (
    TrainingSession,
    TrainingResult,
    RetrieverConfig,
)
from .interfaces import (
    ITrainer,
    IGenerator,
    IRetriever,
    IReranker,
    IClock,
    RAGPipeline,
)

__all__ = [
    # Entities
    "TrainingSession",
    "TrainingResult",
    "RetrieverConfig",
    # Interfaces (contracts)
    "ITrainer",
    "IGenerator",
    "IRetriever",
    "IReranker",
    "IClock",
    "RAGPipeline",
]
