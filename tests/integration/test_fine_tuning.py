import os
import pytest
from quantumgenerator.infrastructure.fine_tuning import (
    LoRATrainer,
    load_data,
    load_model,
    evaluate
)
from quantumgenerator.domain.entities import (
    TrainingSession
)


pytestmark = pytest.mark.integration


def test_load_data():
    with pytest.raises(ValueError):
        load_data("no_thing")
    
    dataset = load_data("openclaw_quantum")
    assert len(dataset) > 0 

def test_load_model():
    loader = load_model("codegemma","google/codegemma-2b")
    assert loader is not None


class TestLoRATrainer():
    def test_train(self):
        session = TrainingSession(
            model_name="google/codegemma-2b",
            data_id="openclaw_quantum",
            output_path="./checkpoints",
            parameter={
                "model_type":"codegemma",
                "per_device_train_batch_size":4,
                "max_steps":100,
                "lora_task_type":"CAUSAL_LM",
                "lora_r":64, 
                "lora_alpha":16, 
                "lora_dropout":0.1
            }
        )
        
        trainer = LoRATrainer()
        trainer.train(session)
        
        assert os.path.exists(os.path.join(
            "models", session.model_name
        ))


def test_evaluate():
    session = TrainingSession(
        model_name="google/codegemma-2b",
        data_id="openclaw_quantum",
        output_path="./checkpoints",
        parameter={
            "model_type":"codegemma",
            "per_device_train_batch_size":4,
            "max_steps":100,
            "lora_task_type":"CAUSAL_LM",
            "lora_r":64, 
            "lora_alpha":16, 
            "lora_dropout":0.1
        }
    )
        
    trainer = LoRATrainer()
    result = trainer.train(session)
    trained_model = trainer.load_model(session, result)

    output = evaluate(
        user_prompt="implement a 3-qubit phase estimation", 
        model=trained_model
    )
    print(output)
    assert len(output) > 0
