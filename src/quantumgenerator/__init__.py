"""QuantumGenerator - Clean Architecture Quantum Code Generation System."""

# Domain layer
from quantumgenerator.domain.entities import (
    RetrieverConfig,
    TrainingSession,
    TrainingResult,
)
from quantumgenerator.domain.interfaces import (
    IGenerator,
    IRetriever,
    IReranker,
    ITrainer,
    # IRepository,
    IClock,
)

# Application layer
from quantumgenerator.application.use_cases import GenerateQuantumCodeUseCase
from quantumgenerator.application.services import CodeGenerationService
from quantumgenerator.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse,
)

# Interfaces (API) layer
from quantumgenerator.interfaces import (
    HEALTHY_STATUS,
    VERSION,
    SERVICE_NAME,
)
from quantumgenerator.interfaces.api import (
    GenerationRequest,
    GenerationResponse,
    router,
)


VERSION = "0.1.0"
AUTHOR = "Huy Pham"
EMAIL = "huypham0297@gmail.com"
__version__ = VERSION
__author__ = AUTHOR
__email__ = EMAIL

__all__ = [
    # Domain
    "RetrieverConfig",
    "TrainingSession",
    "TrainingResult",
    "IGenerator",
    "IRetriever",
    "IReranker",
    "ITrainer",
    # "IRepository",
    "IClock",
    # Application
    "GenerateQuantumCodeUseCase",
    "CodeGenerationService",
    "GenerateQuantumCodeRequest",
    "GenerateQuantumCodeResponse",
    # Interfaces
    "HEALTHY_STATUS",
    "VERSION",
    "SERVICE_NAME",
    "GenerationRequest",
    "GenerationResponse",
    "router",
    "VERSION",
    "AUTHOR",
    "EMAIL",
]
