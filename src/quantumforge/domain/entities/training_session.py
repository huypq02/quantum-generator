from dataclasses import dataclass

@dataclass
class TrainingSession:
    """
    Entities of TrainingSession
    """
    model_name: str
    data_id: str
    output_path: str
    parameter: dict
