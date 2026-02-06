from datasets import load_dataset


def load_data():
    return load_dataset("webxos/OPENCLAW_quantum_dataset", split="train")
