import numpy as np
from src.runtime.pipeline import process_frame, EmotionEvent
from src.runtime.smoother import EmotionSmoother

LABELS = ['anger', 'fear', 'happy', 'sad', 'surprise']


class _FakeDetector:
    def __init__(self, crops):
        self._crops = crops

    def detect_and_crop(self, image, size=112):
        return self._crops


class _FakeEngine:
    labels = LABELS
    input_size = 112

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def infer(self, face):
        return self._probs


def test_process_frame_emits_event():
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    det = _FakeDetector([((10, 10, 80, 80), face)])
    eng = _FakeEngine([0.02, 0.02, 0.9, 0.03, 0.03])
    sm = EmotionSmoother(num_classes=5, alpha=0.0, threshold=0.5)  # alpha=0 直接取当前帧
    ev = process_frame(np.zeros((200, 200, 3), np.uint8), det, eng, sm)
    assert isinstance(ev, EmotionEvent)
    assert ev.label == 'happy'
    assert ev.box == (10, 10, 80, 80)


def test_process_frame_no_face_returns_none():
    det = _FakeDetector([])
    eng = _FakeEngine([0.2] * 5)
    sm = EmotionSmoother(num_classes=5)
    assert process_frame(np.zeros((10, 10, 3), np.uint8), det, eng, sm) is None


def test_process_frame_picks_largest_face():
    small = np.zeros((112, 112, 3), np.uint8)
    big = np.zeros((112, 112, 3), np.uint8)
    det = _FakeDetector([((0, 0, 20, 20), small), ((0, 0, 100, 100), big)])
    eng = _FakeEngine([0.02, 0.02, 0.9, 0.03, 0.03])
    sm = EmotionSmoother(num_classes=5, alpha=0.0, threshold=0.5)
    ev = process_frame(np.zeros((200, 200, 3), np.uint8), det, eng, sm)
    assert ev.box == (0, 0, 100, 100)
