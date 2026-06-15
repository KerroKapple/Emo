import os
import tempfile

import torch
from PIL import Image

import src.config as config
from src.model import get_model
from src.transforms import build_transform
from src.inference import save_checkpoint, load_model_from_checkpoint, predict_probs, resolve_device


def test_resolve_device_cpu():
    assert resolve_device('cpu').type == 'cpu'


def test_checkpoint_roundtrip_and_predict():
    model = get_model('cnn', num_classes=config.NUM_CLASSES, pretrained=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'best_model_cnn.pth')
        save_checkpoint(model, 'cnn', path, val_acc=88.0, epoch=3)

        loaded, info = load_model_from_checkpoint(path, device='cpu')
        assert info['model_type'] == 'cnn'
        assert info['accuracy'] == 88.0
        assert info['input_size'] == config.CNN_INPUT_SIZE
        assert info['classes'] == config.CLASSES

        img = Image.new('RGB', (60, 60), (120, 120, 120))
        probs = predict_probs(loaded, img, build_transform('cnn'), torch.device('cpu'))
        assert len(probs) == config.NUM_CLASSES
        assert abs(sum(probs) - 1.0) < 1e-4


def test_load_missing_checkpoint_raises():
    try:
        load_model_from_checkpoint('does_not_exist.pth', device='cpu')
        assert False, '应抛出 FileNotFoundError'
    except FileNotFoundError:
        pass
