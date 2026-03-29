from quantumgenerator.application.dto import (
    GenerateQuantumCodeRequest,
    GenerateQuantumCodeResponse,
)


def test_request_and_response_dtos() -> None:
    request = GenerateQuantumCodeRequest(query="build bell state")
    response = GenerateQuantumCodeResponse(result="print('ok')", execution_time=0.5)

    assert request.query == "build bell state"
    assert request.retriever_config is None
    assert response.result == "print('ok')"
    assert response.execution_time == 0.5
