from quantumgenerator.domain.interfaces.generator import IGenerator


def evaluate(user_prompt: str, model: IGenerator):
    try:
        text_generation = model.generate(user_prompt)

        return text_generation
    except Exception as e:
        print(f"An unexpected error occurred while evaluating the model: {e}")
        raise RuntimeError("An unexpected error occurred while evaluating the model.")
