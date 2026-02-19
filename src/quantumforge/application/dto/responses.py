from dataclasses import dataclass


@dataclass
class GenerateQuantumCodeResponse:
    """Response DTO for code generation service."""
    result: str
    execution_time: float
