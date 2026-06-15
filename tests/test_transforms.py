from PIL import Image

import src.config as config
from src.transforms import build_transform
from src.model import get_input_size


def test_eval_transform_matches_model_input_size():
    img = Image.new('RGB', (300, 200), (10, 20, 30))
    for model_type in ('cnn', 'resnet18', 'mobilenet'):
        tensor = build_transform(model_type, train=False)(img)
        size = get_input_size(model_type)
        assert tensor.shape == (3, size, size)


def test_train_transform_outputs_correct_size():
    img = Image.new('RGB', (100, 100), (200, 10, 10))
    tensor = build_transform('cnn', train=True)(img)
    assert tensor.shape == (3, config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)
