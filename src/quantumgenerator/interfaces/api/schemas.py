from pydantic import BaseModel


class GenerationRequest(BaseModel):
    query: str

class GenerationResponse(BaseModel):
    result: str
    execution_time: float
