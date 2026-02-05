from datasets import load_dataset


def load_dataset():
    return load_dataset("webxos/OPENCLAW_quantum_dataset", split="train")
