import numpy as np
from src.face.detector import FaceDetector


class _FakeYuNet:
    """模拟 cv2.FaceDetectorYN：返回一行检测 [x,y,w,h, 10 landmarks, score]"""

    def setInputSize(self, size):
        self._size = size

    def detect(self, image):
        row = np.array([30, 40, 60, 60] + [0] * 10 + [0.9], dtype=np.float32)
        return 1, row[None, :]


def test_detect_parses_boxes():
    det = FaceDetector(detector=_FakeYuNet())
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    boxes = det.detect(img)
    assert boxes == [(30, 40, 60, 60)]


def test_detect_and_crop_returns_face():
    det = FaceDetector(detector=_FakeYuNet())
    img = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    out = det.detect_and_crop(img, size=112)
    assert len(out) == 1
    box, face = out[0]
    assert box == (30, 40, 60, 60)
    assert face.shape == (112, 112, 3)


def test_no_face_returns_empty():
    class _Empty(_FakeYuNet):
        def detect(self, image):
            return 0, None
    det = FaceDetector(detector=_Empty())
    assert det.detect((np.random.rand(50, 50, 3) * 255).astype(np.uint8)) == []
