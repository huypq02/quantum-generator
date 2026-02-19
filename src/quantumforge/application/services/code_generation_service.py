from src.quantumforge.domain.interfaces.clock import IClock
from src.quantumforge.application.use_cases import GenerateQuantumCodeUseCase
from src.quantumforge.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse
)


class CodeGenerationService:
    def __init__(
            self, 
            generate_use_case: GenerateQuantumCodeUseCase,
            clock: IClock
    ):
        self.use_case = generate_use_case
        self.clock = clock
    
    def generate(
            self, request: GenerateQuantumCodeRequest
    ) -> GenerateQuantumCodeResponse:
        """
        Code generation service.
        
        :param request: Request DTO for code generation service
        :type request: GenerateQuantumCodeRequest
        :return: Response DTO for code generation service
        :rtype: GenerateQuantumCodeResponse
        """
        start = self.clock.current_utc_timestamp()
        response = self.use_case.execute(request)
        response.execution_time = self.clock.current_utc_timestamp() - start
        
        return response
