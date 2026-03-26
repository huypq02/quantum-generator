from dataclasses import dataclass

@dataclass
class TrainingResult:
    """
    Entity representing the result of a training session.
    
    :param adapter_path: Path to the saved adapter model.
    :type adapter_path: str
    :param model_name: Name of the trained model.
    :type model_name: str
    """
    adapter_path: str
    model_name: str
