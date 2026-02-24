from pydantic import BaseModel


class GenerationRequest(BaseModel):
    """
    Request schema for code generation API.
    
    :param query: User query for code generation.
    :type query: str
    """
    query: str

class GenerationResponse(BaseModel):
    """
    Response schema for code generation API.
    
    :param result: Generated code result.
    :type result: str
    :param execution_time: Time taken to generate the code in seconds.
    :type execution_time: float
    """
    result: str
    execution_time: float
