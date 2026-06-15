import torch
import pytest

import src.config as config
from src.model import get_input_size, get_model, ALL_MODEL_TYPES


def test_get_input_size():
    assert get_input_size('cnn') == config.CNN_INPUT_SIZE
    assert get_input_size('resnet18') == config.TRANSFER_INPUT_SIZE
    assert get_input_size('efficientnet') == config.TRANSFER_INPUT_SIZE


def test_get_input_size_unknown_raises():
    with pytest.raises(ValueError):
        get_input_size('nope')


@pytest.mark.parametrize('model_type', ALL_MODEL_TYPES)
def test_forward_at_native_size(model_type):
    size = get_input_size(model_type)
    model = get_model(model_type, num_classes=5, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 5)


def test_cnn_is_input_size_agnostic():
    model = get_model('cnn', num_classes=5, pretrained=False).eval()
    for size in (48, 96, 224):
        with torch.no_grad():
            assert model(torch.randn(1, 3, size, size)).shape == (1, 5)
