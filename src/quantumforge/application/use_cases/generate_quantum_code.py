from src.quantumforge.domain.interfaces import (
    IGenerator,
    IRetriever
)
from src.quantumforge.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse
)


class GenerateQuantumCodeUseCase:
    """
    Application use case
    """
    def __init__(self, generator: IGenerator, retriever: IRetriever):
        self.generator = generator
        self.retriever = retriever
    
    def execute(
            self, 
            request: GenerateQuantumCodeRequest
    ) -> GenerateQuantumCodeResponse:
        """
        Execute LLMs for generation.
        
        :param request: Request DTO for quantum code generation
        :type request: GenerateQuantumCodeRequest
        :return: Response DTO for quantum code generation
        :rtype: GenerateQuantumCodeResponse
        """
        if not request.query or not request.query.strip():
            return ValueError("Query cannot be empty.")
        
        try:
            context = self.retriever.retrieve(request.query, request.retriever_config)
            prompt = self._compose_prompt(request.query, context)
            result = self.generator.generate(prompt)
        except Exception as e:
            print(f"An unexpected error occurred while generating Quantum code: {e}")
            raise RuntimeError("An unexpected error occurred while generating Quantum code.")
        
        return GenerateQuantumCodeResponse(
            result=result
        )
    
    def _compose_prompt(self, query: str, context: str) -> str:
        """
        Compose query and context into a prompt.
        
        :param query: User input
        :type query: str
        :param context: Context from RAG
        :type context: str
        :return: Prompt
        :rtype: str
        """
        return f"Based on this context:\n{context}\n\n\
            Generate quantum code for: {query}"
