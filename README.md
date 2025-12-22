# QuantumForge 🔬⚛️

QuantumForge is an open-source project that leverages state-of-the-art LLMs (Large Language Models) to generate quantum computing code — including Qiskit (Python), PennyLane, Cirq, and QASM 3 syntax — using Retrieval-Augmented Generation (RAG), prompt templates, and model fine-tuning strategies.

This project is ideal for:

- Researchers exploring quantum circuits
- Developers automating quantum programming
- Educators demonstrating quantum logic gates and algorithms

---

> ## ⚠️ 🚧 **PROJECT STATUS: UNDER DEVELOPMENT** 🚧 ⚠️
>
> **THIS PROJECT IS CURRENTLY IN THE PLANNING AND DEVELOPMENT PHASE.**
>
> All features, models, deployment options, and functionalities described in this README are **PLANNED** and may not be fully implemented yet. This documentation serves as a roadmap and design specification.
>
> Please check the repository for actual implementation status before using in production.

---

## 🧰 Features

> ### 📋 **PLANNED FEATURES** - Not all features are implemented yet

- ✅ Generate **Qiskit**, **PennyLane**, **Cirq**, and **QASM 3** code _(Planned)_
- 🤖 Supports **DeepSeek-Coder**, **CodeLlama**, **Qwen**, **CodeGemma**, or any HuggingFace-compatible open-source LLM _(Planned)_
- 🔁 Integrates **RAG pipeline** using LangChain + ChromaDB _(Planned)_
- ⚡ Compatible with **Google Colab T4**, **RunPod**, or **local inference** _(Planned)_
- 📦 API-ready: deploy as a REST endpoint via **Cloud Run** or **FastAPI** _(Planned)_

---

## 🚀 Getting Started

### Requirements

- Python 3.10+
- HuggingFace Transformers
- LangChain
- ChromaDB or FAISS
- PyTorch (with CUDA for GPU inference)

### Installation

```bash
git clone https://github.com/huypq02/quantum-forge.git
cd quantum-forge
pip install -r requirements.txt
```

---

## 🧪 Running Tests

Run all unit tests:

```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

Run a specific test module:

```bash
python -m unittest tests.unit.test_models
```

Run a specific test class:

```bash
python -m unittest tests.unit.test_models.TestDeepSeekModel
```

Run a specific test method:

```bash
python -m unittest tests.unit.test_models.TestDeepSeekModel.test_load_model
```

---

## 🛠️ Models Supported

> ### 🔮 **PLANNED MODEL SUPPORT** - Integration in progress
>
> The models listed below are targeted for integration. Actual support may vary.

| Model                 | Language       | Params | License    | Supports QASM? | Status    |
| --------------------- | -------------- | ------ | ---------- | -------------- | --------- |
| DeepSeek-Coder 6.7B   | English/Coding | 6.7B   | MIT        | ✅             | _Planned_ |
| Qwen2.5-Coder 7B      | Multilingual   | 7B     | Apache 2.0 | ✅             | _Planned_ |
| CodeLlama 7B Instruct | English/Coding | 7B     | Meta LLAMA | ✅             | _Planned_ |
| CodeGemma 7B          | English/Coding | 7B     | Gemma      | ✅             | _Planned_ |

---

## 📦 Deployment Options

> ### 🚀 **PLANNED DEPLOYMENT OPTIONS** - Implementation roadmap
>
> These deployment options are part of the project roadmap and are not yet available.

🌐 [ ] Run in Colab (with quantized 4-bit models) _(Not yet implemented)_

☁️ [ ] Deploy to Cloud Run or Vertex AI _(Not yet implemented)_

🧪 [ ] Streamlit chatbot interface _(Not yet implemented)_

🧠 [ ] Ollama local support _(Coming soon)_

---

## 📌 Example Prompt

**Prompt:**
"Generate QASM 3 code for a 2-qubit Grover's algorithm with measurement."

**Generated Output (QASM 3):**

```qasm
OPENQASM 3;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
h q;
z q[0];
h q[0];
c = measure q;
```

---

## 🤝 Contributions

Pull requests welcome! If you’ve tested new models or improved prompt templates, feel free to open an issue or PR.

---

## 📄 License

MIT License. See LICENSE for details.

---

## ✨ Acknowledgments

Thanks to the open-source LLM community (DeepSeek, Qwen, Meta AI, HuggingFace) for enabling quantum development at scale.

---

Let me know if you'd like this tailored for a specific model (e.g., DeepSeek only), platform (e.g., Colab-only), or you want a live badge + example repo scaffolding.
