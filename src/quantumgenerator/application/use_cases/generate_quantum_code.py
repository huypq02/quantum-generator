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
            context = self.rag_pipeline.compression_retriever(
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
        context_block = context.strip() if context and context.strip() else "No external context provided."

        return (
            "You are an expert software engineer and technical explainer.\n"
            "Use the context when relevant, but do not invent APIs, classes, or facts that are not supported.\n"
            "If the request is quantum-related, produce correct quantum code. If not, produce the best classical solution.\n\n"
            "Context:\n"
            f"{context_block}\n\n"
            "User Request:\n"
            f"{query}\n\n"
            "Response Requirements:\n"
            "1. Provide complete, executable code.\n"
            "2. Include concise comments only for non-obvious logic.\n"
            "3. Add a short explanation of approach and assumptions.\n"
            "4. Mention trade-offs or edge cases when they matter.\n"
            "5. If requirements are ambiguous, state reasonable assumptions clearly.\n"
        )
