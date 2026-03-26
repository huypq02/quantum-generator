from datasets import load_dataset


def load_data(data_id: str):
    """
    Load training dataset by identifier.
    
    :param data_id: Dataset identifier.
    :type data_id: str
    :return: Loaded training dataset.
    :raises ValueError: If data_id is invalid.
    """
    if data_id == "openclaw_quantum":
        return load_dataset("webxos/OPENCLAW_quantum_dataset", split="train")
    else:
        raise ValueError(f"Invalid data_id {data_id}")
