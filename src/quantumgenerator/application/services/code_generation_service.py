from quantumgenerator.domain.interfaces.clock import IClock
from quantumgenerator.application.use_cases import GenerateQuantumCodeUseCase
from quantumgenerator.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse
)


class CodeGenerationService:
    """
    Service for orchestrating code generation with timing.
    """

    def __init__(
            self, 
            generate_use_case: GenerateQuantumCodeUseCase,
            clock: IClock
    ):
        """
        Initialize the code generation service.
        
        :param generate_use_case: Use case for quantum code generation.
        :type generate_use_case: GenerateQuantumCodeUseCase
        :param clock: Clock for tracking execution time.
        :type clock: IClock
        """
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
        result = self.use_case.execute(request)
        
        return GenerateQuantumCodeResponse(
            result=result,
            execution_time=self.clock.current_utc_timestamp() - start
        )
