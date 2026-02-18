from dataclasses import dataclass

# Use case response
@dataclass
class GenerateQuantumCodeResponse:
    """Response DTO for quantum code generation."""
    result: str

# Service response
@dataclass
class CodeGenerationServiceResponse:
    """Response DTO for code generation service."""
    result: GenerateQuantumCodeResponse
    validation_status: str
    execution_time: float
