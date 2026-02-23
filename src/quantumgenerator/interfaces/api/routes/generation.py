from typing import Annotated
from fastapi import Depends, APIRouter, HTTPException
from quantumgenerator.interfaces.api.constants import (
    HEALTHY_STATUS, 
    SERVICE_NAME, 
    API_VERSION,
)
from quantumgenerator.interfaces.api.schemas import (
    GenerationRequest,
    GenerationResponse,
)
from quantumgenerator.interfaces.api.dependencies import (
    DIContainer,
    create_container,
)


router = APIRouter(prefix="/api")

@router.get("/health")
def heath_check():
    return {
        "status": HEALTHY_STATUS,
        "service": SERVICE_NAME,
        "version": API_VERSION,
    }

@router.post("/generation")
async def generate(
    request: GenerationRequest, 
    container: Annotated[DIContainer, Depends(create_container)],
) -> GenerationResponse:
    service = container.get_code_generation_service()
    result = service.generate(request)

    return result
