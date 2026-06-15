import torch

import src.config as config
from src.utils import set_seed
from src.train import compute_class_weights


def test_set_seed_reproducible():
    set_seed(0)
    a = torch.randn(4)
    set_seed(0)
    b = torch.randn(4)
    assert torch.equal(a, b)


class _FakeDataset:
    def __init__(self, dist):
        self._dist = dist

    def get_class_distribution(self):
        return self._dist


def test_class_weights_inverse_frequency():
    dist = {'anger': 10, 'fear': 100, 'happy': 100, 'sad': 100, 'surprise': 100}
    w = compute_class_weights(_FakeDataset(dist), torch.device('cpu'))
    assert w.shape == (config.NUM_CLASSES,)
    # 样本少的类别权重应更大
    assert w[config.CLASS_TO_IDX['anger']] > w[config.CLASS_TO_IDX['happy']]


def test_class_weights_handles_zero_counts():
    dist = {c: 0 for c in config.CLASSES}
    w = compute_class_weights(_FakeDataset(dist), torch.device('cpu'))
    assert torch.isfinite(w).all()
