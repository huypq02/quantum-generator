from dataclasses import dataclass

@dataclass
class TrainingSession:
    """
    Entities of TrainingSession
    """
    model_name: str
    dataset: any
    output_path: str
    parameter: dict
