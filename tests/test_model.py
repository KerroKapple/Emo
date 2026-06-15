import torch
import pytest
from src.model import get_model

ALL_TYPES = ['cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet', 'efficientnet']


@pytest.mark.parametrize('model_type', ALL_TYPES)
def test_forward_output_shape(model_type):
    model = get_model(model_type, num_classes=5, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 5)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        get_model('not_a_model')
