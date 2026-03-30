# QuantumGenerator

QuantumGenerator is a clean-architecture Python project for generating code with LLMs and retrieval-augmented context.

Current implementation includes:

- Model adapters: CodeGemma, DeepSeek, Qwen, CodeLlama
- RAG pipeline: Chroma or FAISS retrievers with reranking
- Fine-tuning: LoRA trainer on OPENCLAW quantum dataset
- API: FastAPI endpoint for code generation

## Architecture

```
src/quantumgenerator/
├── domain/           # Entities and interfaces
├── application/      # DTOs, use cases, services
├── infrastructure/   # Generators, RAG, fine-tuning, config, logging, time
└── interfaces/       # FastAPI entrypoints and schemas
```

High-level flow:

1. API receives a prompt.
2. Application use case fetches context via RAG pipeline.
3. Prompt + context are passed to selected generator.
4. Service returns generated result with execution time.

## Requirements

- Python 3.9+
- Optional GPU for faster inference/training
- `HF_TOKEN` in `.env` for gated Hugging Face models (CodeGemma/CodeLlama)

Example `.env`:

```
HF_TOKEN=hf_xxxxx
```

## Install

```bash
pip install -e ".[dev]"
```

If you only need runtime dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (Python)

### 1) Use a model directly

```python
from quantumgenerator.infrastructure.generators import ModelFactory

model = ModelFactory.create_model(
    "codegemma",
    model_name="google/codegemma-2b",
    quantize=True,
)

result = model.generate("Generate OpenQASM 3 code for a Bell state")
print(result)
```

### 2) Use the application service (RAG + generation)

```python
from quantumgenerator.application.dto import GenerateQuantumCodeRequest
from quantumgenerator.interfaces.api.dependencies import DIContainer

container = DIContainer()
service = container.get_code_generation_service(
    model_type="codegemma",
    model_name="google/codegemma-2b",
)

response = service.generate(
    GenerateQuantumCodeRequest(query="Implement 3-qubit phase estimation")
)

print(response.result)
print(response.execution_time)
```

## Run the API

```bash
uvicorn quantumgenerator.interfaces.api.main:app --reload
```

Endpoints:

- `GET /api/v1/health`
- `POST /api/v1/generation`

Example request:

```json
{
  "query": "Generate OpenQASM 3 code implementing Grover's algorithm"
}
```

## Configuration

RAG defaults are loaded from `config/config.yaml`.

Important fields:

- `retriever.retriever_type`: `chroma` or `faiss`
- `retriever.vectordb_path`: local vector DB path
- `retriever.documents.paths`: PDF sources to ingest
- `retriever.embedder`: embedding model key (for example `minilm-l6`)
- `retriever.search_kwargs`: retrieval settings (`k`, `lambda_mult`)
- `retriever.rerank_model`: cross-encoder reranker model

## Fine-Tuning (LoRA)

Fine-tuning utilities are in `quantumgenerator.infrastructure.fine_tuning`.

Supported dataset key in code: `openclaw_quantum`.

```python
from quantumgenerator.infrastructure.fine_tuning import LoRATrainer
from quantumgenerator.domain.entities import TrainingSession

session = TrainingSession(
    model_name="google/codegemma-2b",
    data_id="openclaw_quantum",
    output_path="./checkpoints",
    parameter={
        "model_type": "codegemma",
        "per_device_train_batch_size": 4,
        "max_steps": 100,
        "lora_task_type": "CAUSAL_LM",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

trainer = LoRATrainer()
result = trainer.train(session)
print(result.adapter_path)
```

## Testing

Run all tests:

```bash
pytest
```

Run only unit tests:

```bash
pytest tests/unit
```

Run integration tests (downloads models/datasets and can be heavy):

```bash
pytest tests/integration -m integration
```

## Docker

```bash
docker build -t quantumgenerator .
docker run --rm -p 8080:8080 quantumgenerator
```

## License

MIT. See `LICENSE`.
