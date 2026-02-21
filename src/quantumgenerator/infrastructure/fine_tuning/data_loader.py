from datasets import load_dataset


def load_data(data_id: str):
    if data_id == "openclaw_quantum":
        return load_dataset("webxos/OPENCLAW_quantum_dataset", split="train")
    else:
        raise ValueError(f"Invalid data_id {data_id}")
