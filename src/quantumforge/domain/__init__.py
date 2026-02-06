"""Domain Layer - Core business logic, entities, and interfaces."""
from .entities import (
    TrainingSession
)
from .interfaces import (
    ITrainer,
)

__all__ = [
    # Entities
    "TrainingSession",
    # Interfaces (contracts)
    "ITrainer"
]
