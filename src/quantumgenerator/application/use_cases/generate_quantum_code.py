from quantumgenerator.domain import (
    IGenerator,
    RAGPipeline,
    RetrieverConfig,
    IClock
)
from quantumgenerator.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse
)


class GenerateQuantumCodeUseCase:
    """
    Application use case for generating quantum code.
    """

    def __init__(
            self, 
            generator: IGenerator,
            retriever_config: RetrieverConfig,
            rag_pipeline: RAGPipeline,
    ):
        """
        Initialize the use case with dependencies.
        
        :param generator: Code generator instance.
        :type generator: IGenerator
        :param retriever: Document retriever instance.
        :type retriever: IRetriever
        """
        self.generator = generator
        self.retriever_config = retriever_config
        self.rag_pipeline = rag_pipeline
    
    def execute(
            self, 
            request: GenerateQuantumCodeRequest,
    ) -> str:
        """
        Execute LLMs for generation.
        
        :param request: Request DTO for quantum code generation
        :type request: GenerateQuantumCodeRequest
        :return: Result for quantum code generation
        :rtype: str
        :raises ValueError: If query is empty.
        :raises RuntimeError: If generation fails.
        """
        if not request.query or not request.query.strip():
            raise ValueError("Query cannot be empty.")
        
        try:
            config = request.retriever_config or self.retriever_config
            context = self.rag_pipeline.get_context(
                query=request.query,
                config=config
            )
            prompt = self._compose_prompt(request.query, context)
            return self.generator.generate(prompt)
        except Exception as e:
            print(f"An unexpected error occurred while generating Quantum code: {e}")
            raise RuntimeError("An unexpected error occurred while generating Quantum code.")
    
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
