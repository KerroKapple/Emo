import inspect
from dataclasses import asdict

import src.config as config
from src.train import train_model


def test_trainconfig_matches_train_model_signature():
    """train_model(**asdict(TrainConfig())) 的契约：字段集合必须与形参一致"""
    cfg_keys = set(asdict(config.TrainConfig()).keys())
    params = set(inspect.signature(train_model).parameters.keys())
    assert cfg_keys == params
