from src.quantumforge.domain.interfaces import (
    IGenerator,
    IRetriever
)


class GenerateQuantumCodeUseCase:
    """
    Application use case
    """
    def __init__(self, generator: IGenerator, retriever: IRetriever):
        self.generator = generator
        self.retriever = retriever
    
    def execute(self, query: str):
        context = self.retriever.retrieve(query)
        result = self.generator.generate(context)
        return result
