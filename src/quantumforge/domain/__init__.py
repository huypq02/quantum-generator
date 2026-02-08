"""Domain Layer - Core business logic, entities, and interfaces."""
from .entities import (
    TrainingSession,
    TrainingResult
)
from .interfaces import (
    ITrainer,
)

__all__ = [
    # Entities
    "TrainingSession",
    "TrainingResult",
    # Interfaces (contracts)
    "ITrainer"
]
