import numpy as np
from src.runtime.smoother import EmotionSmoother


def _onehot(i, n=5, p=0.9):
    v = np.full(n, (1 - p) / (n - 1), dtype=np.float32)
    v[i] = p
    return v


def test_stabilizes_to_sustained_class():
    s = EmotionSmoother(num_classes=5, alpha=0.5, threshold=0.5)
    idx = None
    for _ in range(5):
        idx, _ = s.update(_onehot(2))
    assert idx == 2


def test_single_noisy_frame_does_not_flip():
    s = EmotionSmoother(num_classes=5, alpha=0.7, threshold=0.5, hysteresis=0.15)
    for _ in range(6):
        s.update(_onehot(2))
    idx, _ = s.update(_onehot(0))  # 单帧噪声
    assert idx == 2


def test_sustained_change_flips():
    s = EmotionSmoother(num_classes=5, alpha=0.5, threshold=0.5, hysteresis=0.1)
    for _ in range(6):
        s.update(_onehot(2))
    for _ in range(6):
        idx, _ = s.update(_onehot(0))
    assert idx == 0


def test_below_threshold_stays_none():
    s = EmotionSmoother(num_classes=5, threshold=0.8)
    idx, _ = s.update(np.full(5, 0.2, dtype=np.float32))
    assert idx is None
