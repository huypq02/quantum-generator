from dataclasses import dataclass


@dataclass
class GenerateQuantumCodeResponse:
    """
    Response DTO for code generation service.
    
    :param result: Generated code output.
    :type result: str
    :param execution_time: Time taken for code generation in seconds.
    :type execution_time: float
    """
    result: str
    execution_time: float
