import pytest
import yaml

@pytest.fixture
def default_config():
    with open("examples/dataset-2/nlu-config.yml") as fp:
        config = yaml.safe_load(fp)
    return config, "examples/dataset-2/"

def test_training_config(default_config, mocker):
    config, config_dir = default_config
    mocker.patch('botshot_nlu.intent.neural_net_model.NeuralNetModel')
    from botshot_nlu.config import TrainingHelper
    helper = TrainingHelper(config, config_dir=config_dir)
    helper.train_intent()
