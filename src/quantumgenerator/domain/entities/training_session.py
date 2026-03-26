from dataclasses import dataclass

@dataclass
class TrainingSession:
    """
    Entity representing a training session configuration.
    
    :param model_name: Name or path of the model to train.
    :type model_name: str
    :param data_id: Identifier for the training dataset.
    :type data_id: str
    :param output_path: Path where training outputs will be saved.
    :type output_path: str
    :param parameter: Dictionary of training parameters.
    :type parameter: dict
    """
    model_name: str
    data_id: str
    output_path: str
    parameter: dict
