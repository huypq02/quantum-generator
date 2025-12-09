# QuantumForge 🔬⚛️

QuantumForge is an open-source project that leverages state-of-the-art LLMs (Large Language Models) to generate quantum computing code — including Qiskit (Python), PennyLane, Cirq, and QASM 3 syntax — using Retrieval-Augmented Generation (RAG), prompt templates, and model fine-tuning strategies.

This project is ideal for:

- Researchers exploring quantum circuits
- Developers automating quantum programming
- Educators demonstrating quantum logic gates and algorithms

---

## 🧰 Features

- ✅ Generate **Qiskit**, **PennyLane**, **Cirq**, and **QASM 3** code
- 🤖 Supports **DeepSeek-Coder**, **CodeLlama**, **Qwen**, or any HuggingFace-compatible open-source LLM
- 🔁 Integrates **RAG pipeline** using LangChain + ChromaDB
- ⚡ Compatible with **Google Colab T4**, **RunPod**, or **local inference**
- 📦 API-ready: deploy as a REST endpoint via **Cloud Run** or **FastAPI**

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

## 🛠️ Models Supported

| Model                 | Language       | Params | License    | Supports QASM? |
| --------------------- | -------------- | ------ | ---------- | -------------- |
| DeepSeek-Coder 6.7B   | English/Coding | 6.7B   | MIT        | ✅             |
| Qwen1.5-Coder 7B      | Multilingual   | 7B     | Apache 2.0 | ✅             |
| CodeLlama 7B Instruct | English/Coding | 7B     | Meta LLAMA | ✅             |

---

## 📦 Deployment Options

🌐 [ ] Run in Colab (with quantized 4-bit models)

☁️ [ ] Deploy to Cloud Run or Vertex AI

🧪 [ ] Streamlit chatbot interface

🧠 [ ] Ollama local support (coming soon)

---

## 📌 Example Prompt

**Prompt:**
"Generate QASM 3 code for a 2-qubit Grover's algorithm with measurement."

**Generated Output (QASM 3):**

```qasm
OPENQASM 3;
include "stdgates.inc";
qubit q[2];
bit c[2];
h q;
z q[1];
h q;
measure q -> c;
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
