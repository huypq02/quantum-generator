from quantumgenerator.domain.interfaces.generator import IGenerator


def evaluate(user_prompt: str, model: IGenerator):
    """
    Evaluate a model by generating text from a user prompt.
    
    :param user_prompt: User input prompt for evaluation.
    :type user_prompt: str
    :param model: Model instance to evaluate.
    :type model: IGenerator
    :return: Generated text output.
    :rtype: str
    :raises RuntimeError: If evaluation fails.
    """
    try:
        text_generation = model.generate(user_prompt)

        return text_generation
    except Exception as e:
        print(f"An unexpected error occurred while evaluating the model: {e}")
        raise RuntimeError("An unexpected error occurred while evaluating the model.")
